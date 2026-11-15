"""Explanatory texts shown in the dashboard."""

HIGHLIGHTS = [
    "This dashboard shows Reddit posts flagged by Reddit users as false information in the "
    "comments. It is an archive of the data collected from {period}; it is no longer updated.",
    'The rule-based method of identifying user flags is described in the "Method" tab.',
]

FLAGS = [
    "The sentences identified as informal flags. Use this tab to check whether the matched "
    "sentences really are flags, and to spot repeated phrases that may come from bots.",
]

SUBMISSIONS = [
    "Here you can inspect each flagged post, and find similar posts across Reddit.",
    "You can filter posts by subreddit, type of flag and number of flags in the left side menu.",
]

AGGREGATE_NOTE = 'The "Flag count" filter does not apply to this tab.'

DOMAINS = [
    "Websites linked by the flagged posts of the current selection. A flagged website is not "
    "necessarily unreliable: the flags express what Reddit users thought of a post.",
]

PRIVACY = [
    "No user names are shown. Mentions of users in comments and titles are masked, and posts "
    "that were removed by moderators or deleted by their authors are left out.",
]

METHOD_INTRO = [
    "This dashboard was created by a team at the University of Mannheim, Germany (School of "
    "Social Sciences, Department of Statistics and Methodology).",
    "The dashboard can serve as a tool for researchers, journalists, or fact checkers, by "
    "helping them filter and analyze the content and spread of information on Reddit.",
    "The dashboard is named STROO? (“Is it true?”), as a nod to the Reddit mascot, the alien SNOO.",
]

METHOD_COLLECTION = [
    "Submissions and comments are collected from Reddit via the Pushshift API with the Python "
    "package PSAW.",
    "Collection started in March 2020 and ran daily while the project was active. This "
    "dashboard is an archive of the data from {period}.",
    "Subreddits were chosen to satisfy the following criteria:",
]
SUBREDDIT_CRITERIA = [
    "the primary language is English",
    "minimum of 10,000 subscribers",
    "most submissions share links to external websites, not self posts, images, videos",
    "external websites shared are news websites, not images, videos, NSFW",
    "there is at least one comment per post (on average).",
]
METHOD_COLLECTION_END = [
    "Only submissions with comments are included; only the first 1000 first-level comments "
    "are extracted.",
]

METHOD_FLAGS = [
    "An informal flag is a comment replying to a submission suspected by a Reddit user to "
    "present false information.",
]
PIPELINE_STEPS = [
    "keyword filtering",
    "sentence tokenization",
    "pattern matching",
    "aggregating to submissions",
    "deciding if flag",
]

METHOD_KEYWORDS_INTRO = "A regex pattern is used to extract comments that contain specific words:"
METHOD_KEYWORDS = [
    "FAKE NEWS: fake, false, bogus, falsehood, fabrication",
    "MISINFORMATION: misinformation, disinformation, malinformation",
    "CLICKBAIT: clickbait, misleading, inaccurate, editorialized, sensationalized",
    "UNRELIABLE: unreliable, untrustworthy, unverified",
    "BULLSHIT: bs, bullshit",
    "PROPAGANDA: propaganda",
]

METHOD_PATTERNS = [
    "Comments were split into sentences. Then, we created a set of patterns, based on a "
    "vocabulary and grammar of the informal flag, and checked if they match each sentence.",
    "Patterns are based on keywords, POS tags and sentence dependencies.",
    "21 patterns were used. Here is an example for each:",
]
METHOD_PATTERNS_END = [
    "<This is fake news> would match, but not <Fake news means false information>.",
    "If at least one sentence matches at least one pattern, the comment is considered a flag.",
    "The method is not perfect: about three quarters of matched comments were confirmed as "
    "flags by manual coding, and a little more than half of the manually identified flags are "
    "matched. It favors precision over recall.",
    "Precision of detecting flags is similar to that obtained by manual coding + machine learning.",
    'You can change the "definition" of a flag by using the left side menu.',
]

METHOD_CLUSTERING = [
    "Submission titles were run through the Universal Sentence Encoder to produce sentence "
    "embeddings.",
    "The 512 embedding variables were reduced to five principal components using PCA.",
    "The five components are used in a K-means cluster analysis. You can select the desired "
    "number of clusters (default: 3).",
    "The same five components are used to find posts similar to a given post.",
]

METHOD_END = [
    "The dashboard was built with Python; pattern matching with spaCy; interface and charts "
    "with Dash/Plotly.",
    "The Dash app is built on top of Flask, served using Gunicorn and Nginx.",
]
PAPER_CITATION = (
    "Achimescu, V., & Chachev, P. D. (2021). Raising the Flag: Monitoring User Perceived "
    "Disinformation on Reddit. Information, 12(1), 4."
)
PAPER_URL = "https://doi.org/10.3390/info12010004"
