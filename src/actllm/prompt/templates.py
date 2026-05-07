SYSTEM_PROMPT = """\
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
You must respond with valid JSON only — no prose, no explanation, no markdown fences.
""".strip()

COT_SYSTEM_PROMPT = """\
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
First reason briefly about the person's likely daily pattern, then produce the final JSON schedule.
""".strip()

_CONSTRAINTS = """\
## Hard Constraints
1. The schedule must start with an activity with start time exactly "00:00", followed by some number of activities, with strictly increasing start times.
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Consecutive home, work or education activities should not occur, eg, a work activity should not be immediately followed by another work activity.
4. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
5. Times must be strictly increasing throughout the schedule.
6. The final activity is expected to be "home" and must start before 24:00 and will be assumed to end at 24:00.

## Output Format
Respond with a single JSON array. Each element is a one-key object mapping the activity name to its start time. No other text.

[
  {{"home": "00:00"}},
  {{"work": "08:30"}},
  {{"home": "17:35"}}
]
"""

ZERO_SHOT_TEMPLATE = """\
Generate a realistic 24-hour activity schedule for the following person.

## Person Attributes
{attributes_block}

""" + _CONSTRAINTS + "\n"

FEW_SHOT_TEMPLATE = """\
Generate a realistic 24-hour activity schedule for the following person.

## Examples
{examples_block}

## Person Attributes
{attributes_block}

""" + _CONSTRAINTS + "\n"

COT_ZERO_SHOT_TEMPLATE = """\
Generate a realistic 24-hour activity schedule for the following person.

## Person Attributes
{attributes_block}

## Hard Constraints
1. The schedule must start with an activity with start time exactly "00:00", followed by some number of activities, with strictly increasing start times.
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Consecutive home, work or education activities should not occur, eg, a work activity should not be immediately followed by another work activity.
4. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
5. Times must be strictly increasing throughout the schedule.
6. The final activity is expected to be "home" and must start before 24:00 and will be assumed to end at 24:00.

## Instructions
First write a brief <reasoning> block: consider whether the person works, their likely wake time, \
and any other activities they might do. Then produce the schedule JSON inside a <schedule> block.

<reasoning>
[Your step-by-step reasoning here]
</reasoning>

<schedule>
[
  {{"home": "00:00"}},
  {{"work": "08:30"}},
  {{"home": "17:35"}}
]
</schedule>
"""

COT_FEW_SHOT_TEMPLATE = """\
Generate a realistic 24-hour activity schedule for the following person.

## Examples
{examples_block}

## Person Attributes
{attributes_block}

## Hard Constraints
1. The schedule must start with an activity with start time exactly "00:00", followed by some number of activities, with strictly increasing start times.
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Consecutive home, work or education activities should not occur, eg, a work activity should not be immediately followed by another work activity.
4. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
5. Times must be strictly increasing throughout the schedule.
6. The final activity is expected to be "home" and must start before 24:00 and will be assumed to end at 24:00.

## Instructions
First write a brief <reasoning> block: consider whether the person works, their likely wake time, \
and any other activities they might do. Then produce the schedule JSON inside a <schedule> block.

<reasoning>
[Your step-by-step reasoning here]
</reasoning>

<schedule>
[
  {{"home": "00:00"}},
  {{"work": "08:30"}},
  {{"home": "17:35"}}
]
</schedule>
"""

RETRY_TEMPLATE = """\
Your previous response had the following errors:

{errors_block}

Please generate a corrected 24-hour activity schedule for the same person.

## Person Attributes
{attributes_block}

## Hard Constraints
1. The schedule must start with an activity with start time exactly "00:00", followed by some number of activities, with strictly increasing start times.
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Consecutive home, work or education activities should not occur, eg, a work activity should not be immediately followed by another work activity.
4. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
5. Times must be strictly increasing throughout the schedule.
6. The final activity is expected to be "home" and must start before 24:00 and will be assumed to end at 24:00.

## Output Format
Respond with a single JSON array. Each element is a one-key object mapping the activity name to its start time. No other text.

[
  {{"home": "00:00"}},
  {{"work": "08:30"}},
  {{"home": "17:35"}}
]
"""
