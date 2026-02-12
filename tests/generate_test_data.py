"""
Generate Needle-in-a-Haystack Test Data
========================================
Creates `tests/dummy_data.txt` — a file filled with filler paragraphs
and a single hidden SECRET_KEY_99 placed at the midpoint.

Usage:
    python tests/generate_test_data.py          # default ~500 KB
    python tests/generate_test_data.py --size 1000   # ~1 MB
"""

import argparse
import os

FILLER_PARAGRAPH = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui deserunt mollit anim id est laborum. "
    "Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. "
    "Nullam varius, turpis et commodo pharetra, est eros bibendum elit, "
    "nec luctus magna felis sollicitudin mauris. Integer in mauris eu "
    "nibh euismod gravida. Duis ac tellus et risus vulputate vehicula. "
    "Donec lobortis risus a elit. Etiam tempor. Ut ullamcorper, ligula "
    "ut dictum pharetra, nisi nunc fringilla magna, in commodo elit erat "
    "nec turpis. Ut pharetra augue nec augue. Nam elit agna, endrerit "
    "sit amet, tincidunt ac, viverra sed, nulla.\n\n"
)

SECRET = "SECRET_KEY_99"


def generate(target_kb: int = 500, output_path: str = None):
    """Generate the test file with a hidden key at the midpoint."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "dummy_data.txt")

    target_bytes = target_kb * 1024
    paragraph_bytes = len(FILLER_PARAGRAPH.encode("utf-8"))
    total_paragraphs = target_bytes // paragraph_bytes

    midpoint = total_paragraphs // 2

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(total_paragraphs):
            if i == midpoint:
                f.write(f"\n--- HIDDEN DATA ---\n{SECRET}\n--- END HIDDEN DATA ---\n\n")
            f.write(FILLER_PARAGRAPH)

    file_size = os.path.getsize(output_path)
    print(f"✅ Created {output_path}")
    print(f"   Size: {file_size / 1024:.1f} KB ({file_size:,} bytes)")
    print(f"   Secret key '{SECRET}' placed at paragraph {midpoint} / {total_paragraphs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate needle-in-a-haystack test data")
    parser.add_argument("--size", type=int, default=500, help="Target file size in KB (default: 500)")
    args = parser.parse_args()
    generate(target_kb=args.size)
