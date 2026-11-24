//! Restart-level power-loss test for the atomic commit slot.

use std::{
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
};

use foyer_storage::{SlotCommit, SlotLayout, read_committed};
use tempfile::tempdir;

fn write_at(file: &mut File, offset: usize, bytes: &[u8]) {
    file.seek(SeekFrom::Start(offset as u64)).unwrap();
    file.write_all(bytes).unwrap();
}

#[test]
fn restart_after_header_power_loss_reads_complete_old_payload() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("slot");
    let layout = SlotLayout::with_header_stride(64, 128).unwrap();

    let old_payload = vec![0x33; 72];
    let new_payload = b"new-short".to_vec();

    {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.set_len(layout.slot_len() as u64).unwrap();

        let mut image = vec![0; layout.slot_len()];
        SlotCommit::commit(&mut image, layout, &old_payload).unwrap();
        file.write_all(&image).unwrap();
        file.sync_all().unwrap();

        let commit = SlotCommit::begin(&image, layout, &new_payload).unwrap();

        write_at(&mut file, commit.data_offset(), &new_payload);
        file.sync_all().unwrap();

        // Power fails after only the first part of the new header reaches disk.
        let torn_prefix = commit.header_bytes().len() / 2;
        write_at(&mut file, commit.header_offset(), &commit.header_bytes()[..torn_prefix]);
        file.sync_all().unwrap();
    }

    let mutated = std::fs::read(&path).unwrap();
    let committed = read_committed(&mutated, layout).unwrap().unwrap();
    assert_eq!(committed.payload(), old_payload);

    // If the complete header reaches disk, restart selects the new generation.
    let mut file = OpenOptions::new().read(true).write(true).open(&path).unwrap();
    let image = std::fs::read(&path).unwrap();
    let commit = SlotCommit::begin(&image, layout, &new_payload).unwrap();
    write_at(&mut file, commit.header_offset(), commit.header_bytes());
    file.sync_all().unwrap();
    drop(file);

    let mutated = std::fs::read(&path).unwrap();
    let committed = read_committed(&mutated, layout).unwrap().unwrap();
    assert_eq!(committed.payload(), new_payload);
}
