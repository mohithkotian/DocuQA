from docuqa.question_generation import parse_generated_questions


def test_parse_generated_questions_from_t5_sequence() -> None:
    raw = (
        "<pad> How many legal deliveries form an over? <sep> "
        "What is a no-ball? <sep> How many runs are awarded for a wide? </s>"
    )
    assert parse_generated_questions(raw) == [
        "How many legal deliveries form an over?",
        "What is a no-ball?",
        "How many runs are awarded for a wide?",
    ]


def test_parse_generated_questions_deduplicates_and_discards_malformed_text() -> None:
    raw = (
        "<pad> What is a no-ball? <sep> what is a no-ball? <sep> "
        "A cricket over contains six legal deliveries. <sep> Why? </s>"
    )
    assert parse_generated_questions(raw) == ["What is a no-ball?"]


def test_parse_generated_questions_obeys_limit() -> None:
    raw = "Who bats first? <sep> Who bowls first? <sep> Who fields first?"
    assert len(parse_generated_questions(raw, limit=2)) == 2
