import re

text = """Printed Page: 1 of 1 Subject Code: BCC301
PAPER ID:311222 Roll No:
BTECH
(SEM III) THEORY EXAMINATION 2023-24
CYBER SECURITY
TIME: 3HRS                                                                 M.MARKS: 70 Note: 1. Attempt all Sections. If require any missing data, then choose suitably.
SECTION A
1. Attempt all questions in brief. Q no. | Question | Marks
a. √ | Define Cyber Crime. | 2
b. √ | What is Bot net. | 2
"""

lines = text.split("\n")
questions_data = []

for line in lines:
    line = line.strip()
    if not line: continue
    lower_line = line.lower()
    if any(x in lower_line for x in ["attempt any", "attempt all", "compulsory", "choose any", "answer any"]):
        # print("Skipping due to filter:", line)
        continue
    q_match = re.match(r'^(Q\d+|\d+)\.?\s*(.*)', line, re.IGNORECASE)
    sub_match = re.match(r'^([a-zA-Z]\)|\([a-zA-Z]\)|[a-zA-Z]\.)\s*(.*)', line, re.IGNORECASE)
    if q_match or sub_match:
        if q_match:
            q_text = q_match.group(2)
        else:
            q_text = sub_match.group(2)
        print("MATCHED:", q_text)
        if len(q_text) > 5:
            questions_data.append(q_text)

print(len(questions_data))
