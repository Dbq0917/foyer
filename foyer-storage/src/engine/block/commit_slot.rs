//! Crash-safe fixed-slot commit primitive.
//!
//! A slot owns two header/data banks. A commit writes the inactive data bank
//! first and then writes that bank's header once. Headers are never written
//! twice in one commit. Recovery chooses the valid header with the greatest
//! generation and verifies its payload checksum.
//!
//! This gives the commit the required atomic outcome:
//!
//! * crash before the new header is complete: the previous generation is read;
//! * crash after the new header is complete: the new generation is read;
//! * a torn header or stale header whose data bank was overwritten is ignored.
//!
//! Place the two headers in separate flush units (normally one page each).
//! Otherwise a lower-level read-modify-write of a shared sector could damage
//! both headers while writing one.

use bytes::{Buf, BufMut};
use foyer_common::error::{Error, ErrorKind, Result};

use foyer_common::bits;

use crate::{io::PAGE, serde::Checksummer};

const COMMIT_MAGIC: u64 = 0x464f_5945_5253_4c31; // "FOYERSL1"
const COMMIT_VERSION: u32 = 1;

/// Serialized commit header length.
pub const COMMIT_HEADER_LEN: usize = 48;

/// One of the two alternating slot banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotBank {
    /// The first bank.
    A,
    /// The second bank.
    B,
}

impl SlotBank {
    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::A),
            1 => Ok(Self::B),
            _ => Err(Error::new(ErrorKind::Parse, "invalid commit slot bank").with_context("bank", value)),
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }

    fn other(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

/// Geometry of a fixed commit slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotLayout {
    header_stride: usize,
    bank_capacity: usize,
}

impl SlotLayout {
    /// Create a layout with page-separated headers.
    pub fn new(bank_capacity: usize) -> Result<Self> {
        Self::with_header_stride(PAGE, bits::align_up(PAGE, bank_capacity))
    }

    /// Create a layout with a custom header stride.
    ///
    /// Production use should keep different headers in independent disk flush
    /// units. A small stride is intended for deterministic unit tests.
    pub fn with_header_stride(header_stride: usize, bank_capacity: usize) -> Result<Self> {
        if header_stride < COMMIT_HEADER_LEN {
            return Err(Error::new(
                ErrorKind::Config,
                "commit slot header stride is smaller than the header",
            )
            .with_context("header_stride", header_stride)
            .with_context("header_len", COMMIT_HEADER_LEN));
        }
        if bank_capacity == 0 {
            return Err(Error::new(
                ErrorKind::Config,
                "commit slot bank capacity must be non-zero",
            ));
        }
        Ok(Self {
            header_stride,
            bank_capacity,
        })
    }

    /// Capacity of one data bank.
    pub fn bank_capacity(&self) -> usize {
        self.bank_capacity
    }

    /// Total byte length of the slot image.
    pub fn slot_len(&self) -> usize {
        self.header_stride * 2 + self.bank_capacity * 2
    }

    fn header_offset(self, bank: SlotBank) -> usize {
        bank.as_u8() as usize * self.header_stride
    }

    fn data_offset(self, bank: SlotBank) -> usize {
        self.header_stride * 2 + bank.as_u8() as usize * self.bank_capacity
    }
}

/// A successfully recovered committed payload.
#[derive(Debug, PartialEq, Eq)]
pub struct CommittedSlot<'a> {
    generation: u64,
    bank: SlotBank,
    payload: &'a [u8],
}

impl<'a> CommittedSlot<'a> {
    /// Monotonic generation of the commit.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Bank that stores the commit.
    pub fn bank(&self) -> SlotBank {
        self.bank
    }

    /// Committed payload.
    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommitHeader {
    generation: u64,
    payload_len: u32,
    bank: SlotBank,
    payload_checksum: u64,
    header_checksum: u64,
}

impl CommitHeader {
    fn encode(&self) -> [u8; COMMIT_HEADER_LEN] {
        let mut buf = [0; COMMIT_HEADER_LEN];
        let mut out = &mut buf[..];
        out.put_u64(COMMIT_MAGIC);
        out.put_u32(COMMIT_HEADER_LEN as u32);
        out.put_u32(COMMIT_VERSION);
        out.put_u64(self.generation);
        out.put_u32(self.payload_len);
        out.put_u8(self.bank.as_u8());
        out.put_u8(0);
        out.put_u16(0);
        out.put_u64(self.payload_checksum);

        let header_checksum = Checksummer::checksum64(&buf[..40]);
        (&mut buf[40..48]).put_u64(header_checksum);
        buf
    }

    fn decode(buf: &[u8], layout: SlotLayout, expected_bank: SlotBank) -> Result<Self> {
        if buf.len() < COMMIT_HEADER_LEN {
            return Err(Error::new(ErrorKind::OutOfRange, "commit slot header is too short"));
        }

        let expected_checksum = (&buf[40..48]).get_u64();
        let actual_checksum = Checksummer::checksum64(&buf[..40]);
        if expected_checksum != actual_checksum {
            return Err(Error::new(
                ErrorKind::ChecksumMismatch,
                "commit slot header checksum mismatch",
            ));
        }

        let mut input = &buf[..COMMIT_HEADER_LEN];
        let magic = input.get_u64();
        let header_len = input.get_u32() as usize;
        let version = input.get_u32();
        let generation = input.get_u64();
        let payload_len = input.get_u32() as usize;
        let bank = SlotBank::from_u8(input.get_u8())?;
        let reserved0 = input.get_u8();
        let reserved1 = input.get_u16();
        let payload_checksum = input.get_u64();
        let header_checksum = input.get_u64();

        if magic != COMMIT_MAGIC {
            return Err(Error::new(ErrorKind::MagicMismatch, "commit slot magic mismatch"));
        }
        if header_len != COMMIT_HEADER_LEN || version != COMMIT_VERSION {
            return Err(Error::new(ErrorKind::Parse, "unsupported commit slot header version"));
        }
        if bank != expected_bank {
            return Err(Error::new(ErrorKind::Parse, "commit slot header is in the wrong bank"));
        }
        if reserved0 != 0 || reserved1 != 0 {
            return Err(Error::new(ErrorKind::Parse, "commit slot reserved bytes are not zero"));
        }
        if payload_len > layout.bank_capacity {
            return Err(Error::new(
                ErrorKind::OutOfRange,
                "commit slot payload length exceeds bank capacity",
            ));
        }

        Ok(Self {
            generation,
            payload_len: payload_len as u32,
            bank,
            payload_checksum,
            header_checksum,
        })
    }
}

/// A prepared commit.
///
/// The caller must write data before writing the header. On a real device,
/// persist the data write before exposing the header write. The header is a
/// single write; it is not written twice by [`SlotCommit::commit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotCommit {
    layout: SlotLayout,
    generation: u64,
    bank: SlotBank,
    data_offset: usize,
    header_offset: usize,
    header: [u8; COMMIT_HEADER_LEN],
}

impl SlotCommit {
    /// Prepare the next commit without modifying the slot.
    pub fn begin(buf: &[u8], layout: SlotLayout, payload: &[u8]) -> Result<Self> {
        if buf.len() < layout.slot_len() {
            return Err(Error::new(ErrorKind::OutOfRange, "commit slot buffer is too short")
                .with_context("expected", layout.slot_len())
                .with_context("actual", buf.len()));
        }
        if payload.len() > layout.bank_capacity {
            return Err(
                Error::new(ErrorKind::BufferSizeLimit, "commit slot payload is too large")
                    .with_context("capacity", layout.bank_capacity)
                    .with_context("payload", payload.len()),
            );
        }

        let committed = read_committed(buf, layout)?;
        let generation = match &committed {
            Some(committed) => committed
                .generation
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorKind::Unsupported, "commit slot generation overflow"))?,
            None => 1,
        };
        let bank = match &committed {
            Some(committed) => committed.bank.other(),
            None => SlotBank::A,
        };
        let header = CommitHeader {
            generation,
            payload_len: payload.len() as u32,
            bank,
            payload_checksum: Checksummer::checksum64(payload),
            header_checksum: 0,
        }
        .encode();

        Ok(Self {
            layout,
            generation,
            bank,
            data_offset: layout.data_offset(bank),
            header_offset: layout.header_offset(bank),
            header,
        })
    }

    /// Commit a payload to an in-memory slot image.
    pub fn commit(buf: &mut [u8], layout: SlotLayout, payload: &[u8]) -> Result<Self> {
        let commit = Self::begin(buf, layout, payload)?;
        commit.write_data(buf, payload)?;
        commit.write_header(buf)?;
        Ok(commit)
    }

    /// Generation that will be committed.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Bank that will be written.
    pub fn bank(&self) -> SlotBank {
        self.bank
    }

    /// Offset of the data bank.
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Offset of the header.
    pub fn header_offset(&self) -> usize {
        self.header_offset
    }

    /// Encoded header bytes.
    pub fn header_bytes(&self) -> &[u8; COMMIT_HEADER_LEN] {
        &self.header
    }

    /// Write the payload into the inactive data bank.
    pub fn write_data(&self, buf: &mut [u8], payload: &[u8]) -> Result<()> {
        self.write_data_prefix(buf, payload, payload.len())
    }

    /// Write only a prefix of the payload, useful for deterministic crash tests.
    pub fn write_data_prefix(&self, buf: &mut [u8], payload: &[u8], prefix_len: usize) -> Result<()> {
        if payload.len() > self.layout.bank_capacity || prefix_len > payload.len() {
            return Err(Error::new(ErrorKind::OutOfRange, "invalid commit slot data write"));
        }
        let end = self
            .data_offset
            .checked_add(prefix_len)
            .ok_or_else(|| Error::new(ErrorKind::OutOfRange, "commit slot data offset overflow"))?;
        if buf.len() < end {
            return Err(Error::new(ErrorKind::OutOfRange, "commit slot buffer is too short"));
        }
        buf[self.data_offset..end].copy_from_slice(&payload[..prefix_len]);
        Ok(())
    }

    /// Write the complete header as the commit operation.
    pub fn write_header(&self, buf: &mut [u8]) -> Result<()> {
        self.write_header_prefix(buf, COMMIT_HEADER_LEN)
    }

    /// Write only a prefix of the header, simulating a torn header write.
    pub fn write_header_prefix(&self, buf: &mut [u8], prefix_len: usize) -> Result<()> {
        if prefix_len > COMMIT_HEADER_LEN {
            return Err(Error::new(
                ErrorKind::OutOfRange,
                "commit slot header prefix is too long",
            ));
        }
        let end = self
            .header_offset
            .checked_add(prefix_len)
            .ok_or_else(|| Error::new(ErrorKind::OutOfRange, "commit slot header offset overflow"))?;
        if buf.len() < end {
            return Err(Error::new(ErrorKind::OutOfRange, "commit slot buffer is too short"));
        }
        buf[self.header_offset..end].copy_from_slice(&self.header[..prefix_len]);
        Ok(())
    }
}

/// Read the latest complete committed payload from a slot image.
pub fn read_committed<'a>(buf: &'a [u8], layout: SlotLayout) -> Result<Option<CommittedSlot<'a>>> {
    if buf.len() < layout.slot_len() {
        return Err(Error::new(ErrorKind::OutOfRange, "commit slot buffer is too short")
            .with_context("expected", layout.slot_len())
            .with_context("actual", buf.len()));
    }

    let mut best: Option<CommittedSlot<'a>> = None;

    for bank in [SlotBank::A, SlotBank::B] {
        let header_offset = layout.header_offset(bank);
        let header = match CommitHeader::decode(&buf[header_offset..], layout, bank) {
            Ok(header) => header,
            Err(_) => continue,
        };

        let data_offset = layout.data_offset(bank);
        let payload_len = header.payload_len as usize;
        let data_end = data_offset + payload_len;
        if data_end > buf.len() {
            continue;
        }
        let payload = &buf[data_offset..data_end];
        if Checksummer::checksum64(payload) != header.payload_checksum {
            continue;
        }

        let candidate = CommittedSlot {
            generation: header.generation,
            bank,
            payload,
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.generation > current.generation)
        {
            best = Some(candidate);
        }
    }

    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_layout() -> SlotLayout {
        SlotLayout::with_header_stride(COMMIT_HEADER_LEN, 128).unwrap()
    }

    #[test]
    fn header_alternates_without_writing_both_headers() {
        let layout = test_layout();
        let mut slot = vec![0; layout.slot_len()];

        let first = SlotCommit::commit(&mut slot, layout, b"first").unwrap();
        assert_eq!(first.generation(), 1);
        assert_eq!(first.bank(), SlotBank::A);
        assert_eq!(read_committed(&slot, layout).unwrap().unwrap().payload(), b"first");

        let second = SlotCommit::commit(&mut slot, layout, b"second-payload").unwrap();
        assert_eq!(second.generation(), 2);
        assert_eq!(second.bank(), SlotBank::B);
        assert_eq!(
            read_committed(&slot, layout).unwrap().unwrap().payload(),
            b"second-payload"
        );

        let third = SlotCommit::commit(&mut slot, layout, b"x").unwrap();
        assert_eq!(third.generation(), 3);
        assert_eq!(third.bank(), SlotBank::A);
        assert_eq!(read_committed(&slot, layout).unwrap().unwrap().payload(), b"x");
    }

    #[test]
    fn power_loss_during_header_write_keeps_previous_complete_payload() {
        let layout = test_layout();
        let mut stable = vec![0; layout.slot_len()];
        let old_payload = vec![0x5a; 64];
        SlotCommit::commit(&mut stable, layout, &old_payload).unwrap();

        let new_payload = b"new".to_vec();
        let prepared = SlotCommit::begin(&stable, layout, &new_payload).unwrap();

        for header_prefix in 0..=COMMIT_HEADER_LEN {
            let mut crashed = stable.clone();
            prepared.write_data(&mut crashed, &new_payload).unwrap();
            prepared.write_header_prefix(&mut crashed, header_prefix).unwrap();

            let committed = read_committed(&crashed, layout).unwrap().unwrap();
            if header_prefix == COMMIT_HEADER_LEN {
                assert_eq!(committed.generation(), 2);
                assert_eq!(committed.payload(), new_payload);
            } else {
                assert_eq!(committed.generation(), 1);
                assert_eq!(committed.payload(), old_payload);
                assert_ne!(committed.payload().len(), 67);
            }
        }
    }

    #[test]
    fn power_loss_during_data_write_keeps_previous_generation() {
        let layout = test_layout();
        let mut stable = vec![0; layout.slot_len()];
        SlotCommit::commit(&mut stable, layout, &[0x11; 64]).unwrap();
        SlotCommit::commit(&mut stable, layout, b"previous-complete-generation").unwrap();

        let new_payload = b"new-short".to_vec();
        let prepared = SlotCommit::begin(&stable, layout, &new_payload).unwrap();
        assert_eq!(prepared.bank(), SlotBank::A);

        for data_prefix in 0..=new_payload.len() {
            let mut crashed = stable.clone();
            prepared
                .write_data_prefix(&mut crashed, &new_payload, data_prefix)
                .unwrap();

            let committed = read_committed(&crashed, layout).unwrap().unwrap();
            assert_eq!(committed.generation(), 2);
            assert_eq!(committed.payload(), b"previous-complete-generation");
        }
    }

    #[test]
    fn corrupted_inactive_header_falls_back_to_valid_bank() {
        let layout = test_layout();
        let mut slot = vec![0; layout.slot_len()];
        SlotCommit::commit(&mut slot, layout, b"a").unwrap();
        SlotCommit::commit(&mut slot, layout, b"b").unwrap();

        let offset = layout.header_offset(SlotBank::B) + 24;
        slot[offset] ^= 0xff;

        let committed = read_committed(&slot, layout).unwrap().unwrap();
        assert_eq!(committed.generation(), 1);
        assert_eq!(committed.payload(), b"a");
    }

    #[test]
    fn rejects_payload_longer_than_bank() {
        let layout = test_layout();
        let slot = vec![0; layout.slot_len()];
        let err = SlotCommit::begin(&slot, layout, &vec![0; layout.bank_capacity() + 1]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::BufferSizeLimit);
    }
}
