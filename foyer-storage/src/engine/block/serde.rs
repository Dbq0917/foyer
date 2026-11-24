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

use std::sync::atomic::AtomicU64;

use bytes::{Buf, BufMut};
use foyer_common::error::{Error, ErrorKind, Result};

use crate::compress::Compression;

const ENTRY_MAGIC: u32 = 0x97_03_27_00;
const ENTRY_MAGIC_MASK: u32 = 0xFF_FF_FF_00;

pub type Sequence = u64;
pub type AtomicSequence = AtomicU64;

#[derive(Debug, PartialEq, Eq)]
pub struct EntryHeader {
    pub key_len: u32,
    pub value_len: u32,
    pub hash: u64,
    pub sequence: Sequence,
    pub checksum: u64,
    pub compression: Compression,
}

impl EntryHeader {
    pub const fn serialized_len() -> usize {
        4 + 4 + 8 + 8 + 8 + 4 /* magic & compression */
    }

    pub fn write(&self, mut buf: impl BufMut) {
        buf.put_u32(self.key_len);
        buf.put_u32(self.value_len);
        buf.put_u64(self.hash);
        buf.put_u64(self.sequence);
        buf.put_u64(self.checksum);

        let v = ENTRY_MAGIC | self.compression.to_u8() as u32;
        buf.put_u32(v);
    }

    pub fn read(mut buf: impl Buf) -> Result<Self> {
        let key_len = buf.get_u32();
        let value_len = buf.get_u32();
        let hash = buf.get_u64();
        let sequence = buf.get_u64();
        let checksum = buf.get_u64();

        let v = buf.get_u32();

        tracing::trace!(
            "read entry header, key len: {key_len}, value_len: {value_len}, hash: {hash}, sequence: {sequence}, checksum: {checksum}, extra: {v}"
        );

        let magic = v & ENTRY_MAGIC_MASK;
        if magic != ENTRY_MAGIC {
            return Err(Error::new(ErrorKind::MagicMismatch, "entry header magic mismatch")
                .with_context("expected", ENTRY_MAGIC)
                .with_context("get", magic));
        }
        let compression = Compression::try_from(v as u8)?;

        Ok(Self {
            key_len,
            value_len,
            hash,
            sequence,
            checksum,
            compression,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serde::{Checksummer, EntryDeserializer};

    #[test]
    fn stale_header_does_not_return_concatenated_tail() {
        let old_value = vec![0xa5; 64];
        let old_key = vec![0x5a; 8];
        let mut payload = Vec::new();
        payload.extend_from_slice(&old_value);
        payload.extend_from_slice(&old_key);

        let header = EntryHeader {
            key_len: old_key.len() as u32,
            value_len: old_value.len() as u32,
            hash: 7,
            sequence: 9,
            checksum: Checksummer::checksum64(&payload),
            compression: Compression::None,
        };

        let mut buf = vec![0; EntryHeader::serialized_len() + payload.len()];
        header.write(&mut buf[..EntryHeader::serialized_len()]);
        buf[EntryHeader::serialized_len()..].copy_from_slice(&payload);

        // A shorter replacement overwrites the prefix, but power loss prevents
        // the header from being committed. The stale header must never produce
        // a value made from the new prefix plus the old tail.
        let new_value = vec![0x11; 3];
        let new_key = vec![0x22; 1];
        let header_len = EntryHeader::serialized_len();
        buf[header_len..header_len + new_value.len()].copy_from_slice(&new_value);
        buf[header_len + new_value.len()..header_len + new_value.len() + new_key.len()].copy_from_slice(&new_key);

        let stale_header = EntryHeader::read(&buf[..EntryHeader::serialized_len()]).unwrap();
        let err = EntryDeserializer::deserialize::<Vec<u8>, Vec<u8>>(
            &buf[EntryHeader::serialized_len()..],
            stale_header.key_len as _,
            stale_header.value_len as _,
            stale_header.compression,
            Some(stale_header.checksum),
        )
        .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::ChecksumMismatch);
    }
}
