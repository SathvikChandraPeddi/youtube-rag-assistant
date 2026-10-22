# This dataset is designed for the video:
# https://www.youtube.com/watch?v=dY7nvNtu_ZA
# (Australia v Bangladesh | First ODI from cricket.com.au)
# Update expected_answer fields for any different video.

EVAL_DATASET = [
    # Case 1: Factual question about a specific thing mentioned
    {
        "question": "What total score was Bangladesh dismissed for in the match?",
        "expected_answer": (
            "Bangladesh was dismissed for 166 runs before Australia began their chase."
        ),
        "relevant_keywords": ["166", "bangladesh", "dismissed"],
        "notes": "Factual score question answerable from middle chunks of transcript.",
    },
    # Case 2: Question asking for an explanation of something discussed
    {
        "question": "How did Phoebe Litchfield perform during Australia's run chase?",
        "expected_answer": (
            "Phoebe Litchfield had a magnificent batting performance, scoring a half-century "
            "to bring up her 12th ODI fifty before being caught out."
        ),
        "relevant_keywords": ["phoebe litchfield", "50", "fifty", "innings"],
        "notes": "Explanation of batting innings answerable from chunk 5.",
    },
    # Case 3: Question about a person, place, or event in the video
    {
        "question": "Where was this match played and what milestone streak did Australia achieve?",
        "expected_answer": (
            "The match took place at Allan Border Field in Brisbane, where the Australian team "
            "extended their unbeaten streak to 22 consecutive one-day international wins."
        ),
        "relevant_keywords": ["brisbane", "allan border field", "22", "consecutive"],
        "notes": "Venue and streak milestone mentioned in opening and closing chunks.",
    },
    # Case 4: Follow-up conversational question 1
    {
        "question": "What did they say about Annabelle Sutherland?",
        "expected_answer": (
            "Commentators noted that Annabelle Sutherland bowled effectively to take multiple wickets, "
            "and later struck the winning runs with power and placement to seal the match."
        ),
        "relevant_keywords": ["annabelle sutherland", "wickets", "win", "bowling"],
        "notes": "Follow-up question covering Sutherland's all-round performance.",
    },
    # Case 5: Follow-up conversational question 2
    {
        "question": "Can you elaborate on Ellyse Perry's contributions in the game?",
        "expected_answer": (
            "Ellyse Perry took an easy catch at cover to dismiss a Bangladesh batter and later batted "
            "sublimely, pushing back and finding boundary gaps."
        ),
        "relevant_keywords": ["ellyse perry", "catch", "cover", "boundary"],
        "notes": "Conversational follow-up about Perry's fielding and batting.",
    },
    # Case 6: Edge case - partially in transcript
    {
        "question": "Who won the match and what was the official Player of the Match award?",
        "expected_answer": (
            "The transcript confirms Australia won by seven wickets to continue their streak, "
            "but the official Player of the Match award recipient is not mentioned in the audio."
        ),
        "relevant_keywords": ["australia", "seven wicket", "won"],
        "notes": "Edge case with partial presence in transcript testing precision.",
    },
    # Case 7: Edge case - completely outside video topic
    {
        "question": "What is the capital city of France and what are its famous monuments?",
        "expected_answer": (
            "This information is not mentioned in the video transcript as the video covers "
            "a women's ODI cricket match between Australia and Bangladesh."
        ),
        "relevant_keywords": [],
        "notes": "Completely out-of-domain edge case to verify grounding and hallucination rejection.",
    },
    # Case 8: Complex multi-part question
    {
        "question": "What happened first in the match and what was the final conclusion?",
        "expected_answer": (
            "First, Australia opened the match with an early wicket straight to slip, and in the end "
            "Australia chased down Bangladesh's total to record a seven-wicket victory and continue their 22-win streak."
        ),
        "relevant_keywords": ["slip", "wicket", "seven wicket", "22", "win"],
        "notes": "Multi-part chronological question spanning start and conclusion of the video.",
    },
]
