import pytest

from docuqa.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, validate_upload_sizes


def test_upload_limit_is_100_mib_and_preserves_supported_formats() -> None:
    assert MAX_UPLOAD_BYTES == 100 * 1024 * 1024
    assert ALLOWED_EXTENSIONS == {".pdf", ".txt", ".docx", ".md", ".markdown"}


def test_pdf_around_70_mb_is_accepted_by_upload_validation() -> None:
    validate_upload_sizes([("women_test_match.pdf", 72 * 1024 * 1024)])


def test_upload_at_100_mib_is_accepted_but_one_byte_over_is_rejected() -> None:
    validate_upload_sizes([("exact_limit.pdf", MAX_UPLOAD_BYTES)])

    with pytest.raises(ValueError, match=r"File exceeds the 100 MB per-file upload limit"):
        validate_upload_sizes([("too_large.pdf", MAX_UPLOAD_BYTES + 1)])


def test_upload_limit_applies_per_file_and_reports_oversized_names() -> None:
    validate_upload_sizes([
        ("large_a.pdf", 72 * 1024 * 1024),
        ("small.txt", 1024),
    ])

    with pytest.raises(ValueError, match=r"Files exceed the 100 MB per-file upload limit: a.pdf, b.pdf"):
        validate_upload_sizes([
            ("a.pdf", MAX_UPLOAD_BYTES + 1),
            ("b.pdf", MAX_UPLOAD_BYTES + 2),
        ])
