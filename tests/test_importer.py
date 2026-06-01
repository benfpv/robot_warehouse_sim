"""Tests for CSV loading and validation in data.importer.Importer."""
import pytest

from data.importer import Importer


# ---------------------------------------------------------------------------
# init_import_csv_as_list
# ---------------------------------------------------------------------------

def test_import_csv_reads_rows_and_skips_header(tmp_path):
    csv_file = tmp_path / "items.csv"
    csv_file.write_text("name,labels\napple,fruit\norange,fruit\n")
    rows = Importer.init_import_csv_as_list(str(csv_file))
    assert rows == [["apple", "fruit"], ["orange", "fruit"]]


def test_import_csv_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        Importer.init_import_csv_as_list(str(missing))


def test_import_csv_empty_file_raises(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    with pytest.raises(ValueError):
        Importer.init_import_csv_as_list(str(csv_file))


def test_import_csv_header_only_returns_empty(tmp_path):
    csv_file = tmp_path / "header_only.csv"
    csv_file.write_text("name,labels\n")
    assert Importer.init_import_csv_as_list(str(csv_file)) == []


# ---------------------------------------------------------------------------
# init_objectify_items_list
# ---------------------------------------------------------------------------

def test_objectify_items_valid_rows():
    rows = [["apple", "food", "a fruit", "02 07 2022", "1 1 1", "1"]]
    items = Importer.init_objectify_items_list(rows)
    assert "apple" in items


def test_objectify_items_malformed_row_raises():
    rows = [["apple", "food"]]  # too few columns
    with pytest.raises(ValueError):
        Importer.init_objectify_items_list(rows)


# ---------------------------------------------------------------------------
# init_objectify_addresses_list
# ---------------------------------------------------------------------------

def test_objectify_addresses_valid_rows():
    rows = [["1 Main St", "Toronto", "Ontario", "A1A1A1"]]
    addrs = Importer.init_objectify_addresses_list(rows)
    assert "1 Main St" in addrs


def test_objectify_addresses_malformed_row_raises():
    rows = [["1 Main St", "Toronto"]]  # too few columns
    with pytest.raises(ValueError):
        Importer.init_objectify_addresses_list(rows)


# ---------------------------------------------------------------------------
# Real resource files still load
# ---------------------------------------------------------------------------

def test_real_resource_files_load():
    items = Importer.init_objectify_items_list(
        Importer.init_import_csv_as_list("resources/list_items.csv")
    )
    addrs = Importer.init_objectify_addresses_list(
        Importer.init_import_csv_as_list("resources/list_addresses.csv")
    )
    assert items and addrs
