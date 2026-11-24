// Copyright 2026 foyer Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{sync::Arc, time::Instant};

use bytes::{Buf, BufMut};
use foyer_common::{
    bits,
    code::{StorageKey, StorageValue},
    error::ErrorKind,
    metrics::Metrics,
};

use crate::{
    compress::Compression,
    engine::block::serde::{EntryHeader, Sequence},
    io::{
        PAGE,
        bytes::{IoSlice, IoSliceMut},
    },
    serde::{Checksummer, EntrySerializer},
};

/// [`EntryIndex`] index entry in a blob, which can be used to speed up recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobEntryIndex {
    /// Entry hash.
    pub hash: u64,
    /// Entry sequence.
    pub sequence: Sequence,
    /// Byte offset to the block.
    pub offset: u32,
    /// Serialized entry length.
    pub len: u32,
}

impl BlobEntryIndex {
    pub const fn serialized_len() -> usize {
        8 + 8 + 4 + 4
    }

    pub fn write(&self, buf: &mut [u8]) {
        (&mut buf[0..8]).put_u64(self.hash);
        (&mut buf[8..16]).put_u64(self.sequence);
        (&mut buf[16..20]).put_u32(self.offset);
        (&mut buf[20..24]).put_u32(self.len);
    }

    pub fn read(buf: &[u8]) -> Self {
        let hash = (&buf[0..8]).get_u64();
        let sequence = (&buf[8..16]).get_u64();
        let offset = (&buf[16..20]).get_u32();
        let len = (&buf[20..24]).get_u32();
        Self {
            hash,
            sequence,
            offset,
            len,
        }
    }
}

/// # Format
///
/// ```plain
/// | checksum (8B) | count (4B) | index 0 | index 1 | ... | index N |
/// ```
#[derive(Debug)]
pub struct BlobIndex {
    bytes: IoSliceMut,
    count: usize,
}

impl BlobIndex {
    pub const CHECKSUM_BYTES: usize = 8;

    pub const COUNT_OFFSET: usize = 8;
    pub const COUNT_BYTES: usize = 4;

    pub const INDEX_OFFSET: usize = 12;

    pub fn new(bytes: IoSliceMut) -> Self {
        Self { bytes, count: 0 }
    }

    pub fn capacity(&self) -> usize {
        (self.bytes.len() - BlobIndex::INDEX_OFFSET) / BlobEntryIndex::serialized_len()
    }

    pub fn is_full(&self) -> bool {
        self.count >= self.capacity()
    }

    /// # Panics
    ///
    /// Panics if the buffer is already full.
    pub fn write(&mut self, index: &BlobEntryIndex) {
        assert!(!self.is_full());
        let start = BlobIndex::INDEX_OFFSET + self.count * BlobEntryIndex::serialized_len();
        let end = start + BlobEntryIndex::serialized_len();
        index.write(&mut self.bytes[start..end]);
        self.count += 1;
    }

    pub fn seal(&mut self) -> IoSliceMut {
        (&mut self.bytes[BlobIndex::COUNT_OFFSET..BlobIndex::COUNT_OFFSET + BlobIndex::COUNT_BYTES])
            .put_u32(self.count as _);
        let checksum = Checksummer::checksum64(&self.bytes[BlobIndex::CHECKSUM_BYTES..]);
        (&mut self.bytes[0..BlobIndex::CHECKSUM_BYTES]).put_u64(checksum);
        self.bytes.clone()
    }

    pub fn reset(&mut self) {
        self.count = 0;
    }
}

#[derive(Debug)]
pub struct BlobIndexReader;

impl BlobIndexReader {
    pub fn read(buf: &[u8]) -> Option<Vec<BlobEntryIndex>> {
        let checksum = Checksummer::checksum64(&buf[BlobIndex::CHECKSUM_BYTES..]);
        let expected = (&buf[0..BlobIndex::CHECKSUM_BYTES]).get_u64();
        if checksum != expected {
            tracing::trace!(checksum, expected, "[blob index reader]: checksum mismatch");
            return None;
        }
        let count =
            (&buf[BlobIndex::COUNT_OFFSET..BlobIndex::COUNT_OFFSET + BlobIndex::COUNT_BYTES]).get_u32() as usize;
        let indices = buf[BlobIndex::INDEX_OFFSET..BlobIndex::INDEX_OFFSET + count * BlobEntryIndex::serialized_len()]
            .chunks_exact(BlobEntryIndex::serialized_len())
            .map(BlobEntryIndex::read)
            .collect();
        Some(indices)
    }
}

#[derive(Debug, Clone)]
pub struct BufferEntryInfo {
    pub hash: u64,
    pub sequence: Sequence,
    pub offset: usize,
    pub len: usize,
}

#[derive(Debug)]
pub struct Buffer {
    bytes: IoSliceMut,
    written: usize,
    entry_infos: Vec<BufferEntryInfo>,

    max_entry_size: usize,

    metrics: Arc<Metrics>,
}

impl Buffer {
    pub fn new(bytes: IoSliceMut, max_entry_size: usize, metrics: Arc<Metrics>) -> Self {
        Self {
            bytes,
            written: 0,
            entry_infos: vec![],
            max_entry_size,
            metrics,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entry_infos.is_empty()
    }

    pub fn push<K, V>(&mut self, key: &K, value: &V, hash: u64, compression: Compression, sequence: Sequence) -> bool
    where
        K: StorageKey,
        V: StorageValue,
    {
        tracing::trace!(hash, "[blob writer]: push");

        let mut offset = self.written;
        if self.bytes.len().saturating_sub(offset) < EntryHeader::serialized_len() {
            offset = bits::align_up(PAGE, offset);
        }
        if offset + EntryHeader::serialized_len() > self.bytes.len() {
            return false;
        }

        let ser = Instant::now();

        let info = match EntrySerializer::serialize(
            key,
            value,
            compression,
            &mut self.bytes[offset + EntryHeader::serialized_len()..],
        ) {
            Ok(info) => info,
            Err(e) => {
                if e.kind() != ErrorKind::BufferSizeLimit {
                    tracing::warn!(?e, "[blob writer]: serialize entry kv error");
                }
                return false;
            }
        };
        let len = EntryHeader::serialized_len() + info.key_len as usize + info.value_len as usize;

        // Keep small entries within a single page, so packing does not increase read amplification. Entries larger
        // than a page still start at a page boundary and keep the same read behavior as before.
        let start = Self::entry_start(offset, len);
        let aligned = bits::align_up(PAGE, len);
        if aligned > self.max_entry_size || start + len > self.bytes.len() {
            return false;
        }

        if start != offset {
            self.bytes.copy_within(offset..offset + len, start);
        }

        let checksum = Checksummer::checksum64(&self.bytes[start + EntryHeader::serialized_len()..start + len]);
        let header = EntryHeader {
            key_len: info.key_len as _,
            value_len: info.value_len as _,
            hash,
            sequence,
            checksum,
            compression,
        };
        header.write(&mut self.bytes[start..start + EntryHeader::serialized_len()]);

        self.metrics
            .storage_entry_serialize_duration
            .record(ser.elapsed().as_secs_f64());

        let info = BufferEntryInfo {
            hash,
            sequence,
            offset: start,
            len,
        };
        self.entry_infos.push(info);
        self.written = start + len;

        tracing::trace!(hash, "[blob writer]: push finish");

        true
    }

    /// Serialize an serialized kv entry slice into the dest.
    pub fn push_slice(&mut self, slice: &[u8], hash: u64, sequence: Sequence) -> bool {
        tracing::trace!(hash, "[blob writer]: push slice");

        let len = slice.len();
        let aligned = bits::align_up(PAGE, slice.len());
        let start = Self::entry_start(self.written, len);

        if aligned > self.max_entry_size || start + len > self.bytes.len() {
            return false;
        }

        self.bytes[start..start + len].copy_from_slice(slice);

        let info = BufferEntryInfo {
            hash,
            sequence,
            offset: start,
            len,
        };
        self.entry_infos.push(info);
        self.written = start + len;

        tracing::trace!(hash, "[blob writer]: push slice finish");

        true
    }

    fn entry_start(written: usize, len: usize) -> usize {
        if len <= PAGE {
            let page_start = bits::align_down(PAGE, written);
            if written - page_start + len <= PAGE {
                written
            } else {
                page_start + PAGE
            }
        } else {
            bits::align_up(PAGE, written)
        }
    }

    pub fn finish(self) -> (IoSliceMut, Vec<BufferEntryInfo>) {
        tracing::trace!(infos=?self.entry_infos, "[buffer]: finish");
        (self.bytes, self.entry_infos)
    }
}

#[derive(Debug)]
pub struct SplitCtx {
    current_part_blob_offset: usize,
    current_blob_index: BlobIndex,

    current_blob_block_offset: usize,

    block_size: usize,
    blob_index_size: usize,
}

impl SplitCtx {
    pub fn new(block_size: usize, blob_index_size: usize) -> Self {
        Self {
            current_part_blob_offset: blob_index_size,
            current_blob_index: BlobIndex::new(IoSliceMut::new(blob_index_size)),
            current_blob_block_offset: 0,
            block_size,
            blob_index_size,
        }
    }
}

#[derive(Debug)]
pub struct Splitter;

impl Splitter {
    pub fn split(ctx: &mut SplitCtx, bytes: IoSlice, entry_infos: Vec<BufferEntryInfo>) -> Batch {
        let mut batch = Batch {
            blocks: vec![Block { blob_parts: vec![] }],
            bytes: bytes.clone(),
        };

        assert!(
            !ctx.current_blob_index.is_full(),
            "[splitter] Blob index writer cannot be full at this point, it is supposed to be flushed in the last batch."
        );

        let mut indices = vec![];
        let mut part_start = None;
        let mut part_end = 0;

        for info in entry_infos.into_iter() {
            tracing::trace!(?info, "[splitter]: handle entry");

            'handle: loop {
                // Split blob if blob index is full.
                if ctx.current_blob_index.is_full() {
                    Self::seal_part(
                        ctx,
                        &mut batch,
                        &bytes,
                        &mut indices,
                        &mut part_start,
                        &mut part_end,
                        true,
                    );
                    continue 'handle;
                }

                let start = part_start.unwrap_or_else(|| bits::align_down(PAGE, info.offset));
                let part_len = bits::align_up(PAGE, info.offset - start + info.len);

                // Split blob and block if block is full.
                if ctx.current_blob_block_offset + ctx.current_part_blob_offset + part_len > ctx.block_size {
                    Self::seal_part(
                        ctx,
                        &mut batch,
                        &bytes,
                        &mut indices,
                        &mut part_start,
                        &mut part_end,
                        true,
                    );
                    Self::split_block(ctx, &mut batch);
                    continue 'handle;
                }

                if part_start.is_none() {
                    part_start = Some(start);
                }

                let index = BlobEntryIndex {
                    hash: info.hash,
                    sequence: info.sequence,
                    offset: (ctx.current_part_blob_offset + info.offset - start) as u32,
                    len: info.len as u32,
                };

                tracing::trace!(?index, ?info, "[splitter]: append entry");

                ctx.current_blob_index.write(&index);
                indices.push(index);
                part_end = part_end.max(info.offset + info.len);

                break 'handle;
            }
        }

        Self::seal_part(
            ctx,
            &mut batch,
            &bytes,
            &mut indices,
            &mut part_start,
            &mut part_end,
            false,
        );

        batch
    }

    fn seal_part(
        ctx: &mut SplitCtx,
        batch: &mut Batch,
        bytes: &IoSlice,
        indices: &mut Vec<BlobEntryIndex>,
        part_start: &mut Option<usize>,
        part_end: &mut usize,
        force_new_blob: bool,
    ) {
        tracing::trace!("[splitter]: seal part");

        if indices.is_empty() {
            return;
        }

        let start = part_start.take().unwrap();
        let end = bits::align_up(PAGE, *part_end);
        let data = bytes.slice(start..end);
        let index = ctx.current_blob_index.seal();
        let indices = std::mem::take(indices);

        let part = BlobPart {
            blob_block_offset: ctx.current_blob_block_offset,
            index,
            part_blob_offset: ctx.current_part_blob_offset,
            data,
            indices,
        };

        let part_size = part.data.len();
        if force_new_blob || ctx.current_blob_index.is_full() {
            ctx.current_blob_index.reset();
            ctx.current_blob_block_offset += ctx.current_part_blob_offset + part_size;
            ctx.current_part_blob_offset = ctx.blob_index_size;
        } else {
            ctx.current_part_blob_offset += part_size;
        }

        *part_end = 0;

        batch.blocks.last_mut().unwrap().blob_parts.push(part);
    }

    fn split_block(ctx: &mut SplitCtx, batch: &mut Batch) {
        tracing::trace!("[splitter]; split block");
        batch.blocks.push(Block { blob_parts: vec![] });
        ctx.current_blob_block_offset = 0;
        ctx.current_part_blob_offset = ctx.blob_index_size;
        ctx.current_blob_index.reset();
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct BlobPart {
    /// Blob offset to the block.
    pub blob_block_offset: usize,
    pub index: IoSliceMut,
    /// Blob part offset to the blob.
    pub part_blob_offset: usize,
    pub data: IoSlice,

    pub indices: Vec<BlobEntryIndex>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Block {
    pub blob_parts: Vec<BlobPart>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Batch {
    pub blocks: Vec<Block>,
    pub bytes: IoSlice,
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    const KB: usize = 1024;

    #[test_log::test]
    fn test_blob_index_serde() {
        let mut bi = BlobIndex::new(IoSliceMut::new(PAGE * 2));
        let indices = (0..bi.capacity() / 2)
            .map(|i| BlobEntryIndex {
                hash: i as u64,
                sequence: i as u64,
                offset: i as u32,
                len: i as u32,
            })
            .collect_vec();

        indices.iter().take(bi.capacity() / 2).for_each(|index| bi.write(index));
        let buf = bi.seal();
        let ins = BlobIndexReader::read(&buf).unwrap();
        assert_eq!(indices.iter().take(bi.capacity() / 2).cloned().collect_vec(), ins);

        indices.iter().skip(bi.capacity() / 2).for_each(|index| bi.write(index));
        let buf = bi.seal();
        let ins = BlobIndexReader::read(&buf).unwrap();
        assert_eq!(indices, ins);
    }

    #[test_log::test]
    fn test_buffer() {
        const BLOCK_SIZE: usize = 16 * KB;
        const BLOB_INDEX_SIZE: usize = 4 * KB;
        const MAX_ENTRY_SIZE: usize = BLOCK_SIZE - BLOB_INDEX_SIZE;
        const BATCH_SIZE: usize = 64 * KB;

        let mut ctx = SplitCtx::new(BLOCK_SIZE, BLOB_INDEX_SIZE);

        // 1. Test write single blob part.

        let mut buffer = Buffer::new(IoSliceMut::new(BATCH_SIZE), MAX_ENTRY_SIZE, Arc::new(Metrics::noop()));

        // 4K
        assert!(buffer.push(&1u64, &vec![1u8; 3 * KB], 1, Compression::None, 1));

        // 16K (deny)
        assert!(!buffer.push(&2u64, &vec![2u8; 13 * KB], 2, Compression::None, 2));

        // 4K
        assert!(buffer.push(&3u64, &vec![3u8; 3 * KB], 3, Compression::None, 3));

        let (buf, infos) = buffer.finish();
        let buf = buf.into_io_slice();
        let batch = Splitter::split(&mut ctx, buf.clone(), infos.clone());

        assert_eq!(
            batch,
            Batch {
                bytes: buf.clone(),
                blocks: vec![Block {
                    blob_parts: vec![BlobPart {
                        blob_block_offset: 0,
                        index: batch.blocks[0].blob_parts[0].index.clone(),
                        part_blob_offset: BLOB_INDEX_SIZE,
                        data: buf.slice(0..8 * KB),
                        indices: vec![
                            BlobEntryIndex {
                                hash: 1,
                                sequence: 1,
                                offset: BLOB_INDEX_SIZE as u32,
                                len: infos[0].len as u32,
                            },
                            BlobEntryIndex {
                                hash: 3,
                                sequence: 3,
                                offset: BLOB_INDEX_SIZE as u32 + 4 * KB as u32,
                                len: infos[1].len as u32,
                            }
                        ]
                    }]
                }]
            }
        );

        // 2. Test continue to write blob part and block split.

        let mut buffer = Buffer::new(IoSliceMut::new(BATCH_SIZE), MAX_ENTRY_SIZE, Arc::new(Metrics::noop()));

        // 4K, block split
        assert!(buffer.push(&4u64, &vec![4u8; 3 * KB], 4, Compression::None, 4));

        // 8K
        assert!(buffer.push(&5u64, &vec![5u8; 7 * KB], 5, Compression::None, 5));

        // 8K, block early split
        assert!(buffer.push(&6u64, &vec![6u8; 7 * KB], 6, Compression::None, 6));

        let (buf, infos) = buffer.finish();
        let buf = buf.into_io_slice();
        let batch = Splitter::split(&mut ctx, buf.clone(), infos.clone());

        assert_eq!(
            batch,
            Batch {
                bytes: buf.clone(),
                blocks: vec![
                    Block {
                        blob_parts: vec![BlobPart {
                            blob_block_offset: 0,
                            index: batch.blocks[0].blob_parts[0].index.clone(),
                            part_blob_offset: BLOB_INDEX_SIZE + 8 * KB,
                            data: buf.slice(0..4 * KB),
                            indices: vec![BlobEntryIndex {
                                hash: 4,
                                sequence: 4,
                                offset: BLOB_INDEX_SIZE as u32 + 8 * KB as u32,
                                len: infos[0].len as u32,
                            }]
                        }]
                    },
                    Block {
                        blob_parts: vec![BlobPart {
                            blob_block_offset: 0,
                            index: batch.blocks[1].blob_parts[0].index.clone(),
                            part_blob_offset: BLOB_INDEX_SIZE,
                            data: buf.slice(4 * KB..12 * KB),
                            indices: vec![BlobEntryIndex {
                                hash: 5,
                                sequence: 5,
                                offset: BLOB_INDEX_SIZE as u32,
                                len: infos[1].len as u32,
                            }]
                        }]
                    },
                    Block {
                        blob_parts: vec![BlobPart {
                            blob_block_offset: 0,
                            index: batch.blocks[2].blob_parts[0].index.clone(),
                            part_blob_offset: BLOB_INDEX_SIZE,
                            data: buf.slice(12 * KB..20 * KB),
                            indices: vec![BlobEntryIndex {
                                hash: 6,
                                sequence: 6,
                                offset: BLOB_INDEX_SIZE as u32,
                                len: infos[2].len as u32,
                            }]
                        }]
                    }
                ]
            }
        );

        // 3. Test leave first block empty.

        let mut buffer = Buffer::new(IoSliceMut::new(BATCH_SIZE), MAX_ENTRY_SIZE, Arc::new(Metrics::noop()));

        // 8K, block split
        assert!(buffer.push(&7u64, &vec![7u8; 7 * KB], 7, Compression::None, 7));

        let (buf, infos) = buffer.finish();
        let buf = buf.into_io_slice();
        let batch = Splitter::split(&mut ctx, buf.clone(), infos.clone());

        assert_eq!(
            batch,
            Batch {
                bytes: buf.clone(),
                blocks: vec![
                    Block { blob_parts: vec![] },
                    Block {
                        blob_parts: vec![BlobPart {
                            blob_block_offset: 0,
                            index: batch.blocks[1].blob_parts[0].index.clone(),
                            part_blob_offset: BLOB_INDEX_SIZE,
                            data: buf.slice(0..8 * KB),
                            indices: vec![BlobEntryIndex {
                                hash: 7,
                                sequence: 7,
                                offset: BLOB_INDEX_SIZE as u32,
                                len: infos[0].len as u32,
                            }]
                        }]
                    },
                ]
            }
        );
    }

    #[test_log::test]
    fn test_split_block_last_entry() {
        const KB: usize = 1 << 10;

        const BLOCK_SIZE: usize = 64 * KB;
        const BLOB_INDEX_SIZE: usize = 4 * KB;
        const BATCH_SIZE: usize = 128 * KB;

        // Remain 4 KB in size and 1 entry in count.
        let mut ctx = SplitCtx::new(BLOCK_SIZE, BLOB_INDEX_SIZE);
        ctx.current_blob_block_offset = 40 * KB;
        ctx.current_part_blob_offset = 16 * KB;
        ctx.current_blob_index.count = ctx.current_blob_index.capacity() - 1;

        let shared_io_slice = IoSliceMut::new(BATCH_SIZE).into_io_slice();
        let batch = Splitter::split(
            &mut ctx,
            shared_io_slice.clone(),
            vec![
                BufferEntryInfo {
                    hash: 0,
                    sequence: 0,
                    offset: 0,
                    len: 1234,
                },
                BufferEntryInfo {
                    hash: 1,
                    sequence: 1,
                    offset: 4 * KB,
                    len: 1234,
                },
            ],
        );

        println!("{batch:?}");
        assert_eq!(
            batch,
            Batch {
                blocks: vec![
                    Block {
                        blob_parts: vec![BlobPart {
                            blob_block_offset: 40 * KB,
                            index: batch.blocks[0].blob_parts[0].index.clone(),
                            part_blob_offset: 16 * KB,
                            data: shared_io_slice.slice(0..4 * KB),
                            indices: vec![BlobEntryIndex {
                                hash: 0,
                                sequence: 0,
                                offset: 16 * KB as u32,
                                len: 1234,
                            }]
                        }]
                    },
                    Block {
                        blob_parts: vec![BlobPart {
                            blob_block_offset: 0,
                            index: batch.blocks[1].blob_parts[0].index.clone(),
                            part_blob_offset: 4 * KB,
                            data: shared_io_slice.slice(4 * KB..8 * KB),
                            indices: vec![BlobEntryIndex {
                                hash: 1,
                                sequence: 1,
                                offset: 4 * KB as u32,
                                len: 1234,
                            }]
                        }]
                    }
                ],
                bytes: shared_io_slice,
            }
        );
    }
}
