========================================================================
CONFIG: zero_shot.yaml
========================================================================
--- SYSTEM ---
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
You must respond with valid JSON only — no prose, no explanation, no markdown fences.
--- USER ---
Generate a realistic 24-hour activity schedule for the following person.

## Person Attributes
- Age: >64
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 5.76-9.6

## Hard Constraints
1. The schedule MUST start with activity "home" at exactly "00:00".
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
4. Times must be strictly increasing throughout the schedule.
5. The final activity must start before 23:59 and will be assumed to end at 24:00.

## Output Format
Respond with a single JSON object in exactly this structure. No other text.

{
  "schedule": [
    {"activity": "home", "start": "00:00"},
    {"activity": "work", "start": "08:30"},
    {"activity": "home", "start": "17:35"}
  ]
}



========================================================================
CONFIG: few_shot_random.yaml
========================================================================
--- SYSTEM ---
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
You must respond with valid JSON only — no prose, no explanation, no markdown fences.
--- USER ---
Generate a realistic 24-hour activity schedule for the following person.

## Examples
### Example 1
Person:
- Age: 34-49
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: unemployed
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: thursday
- Access/egress distance (km) to public transit: ≤3.2

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}]}

### Example 2
Person:
- Age: 16-34
- Sex: male
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: student
- Annual household income (euros): >60798
- Household location type: suburban
- Day of survey: wednesday
- Access/egress distance (km) to public transit: 9.6-17.2968

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}, {"activity": "education", "start": "07:10"}, {"activity": "visit", "start": "14:30"}, {"activity": "education", "start": "17:30"}, {"activity": "home", "start": "17:45"}]}

### Example 3
Person:
- Age: 16-34
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: other
- Annual household income (euros): 40228-60798
- Household location type: urban
- Day of survey: tuesday
- Access/egress distance (km) to public transit: 3.2-5.76

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}]}

## Person Attributes
- Age: >64
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 5.76-9.6

## Hard Constraints
1. The schedule MUST start with activity "home" at exactly "00:00".
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
4. Times must be strictly increasing throughout the schedule.
5. The final activity must start before 23:59 and will be assumed to end at 24:00.

## Output Format
Respond with a single JSON object in exactly this structure. No other text.

{
  "schedule": [
    {"activity": "home", "start": "00:00"},
    {"activity": "work", "start": "08:30"},
    {"activity": "home", "start": "17:35"}
  ]
}



========================================================================
CONFIG: few_shot_stratified.yaml
========================================================================
--- SYSTEM ---
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
You must respond with valid JSON only — no prose, no explanation, no markdown fences.
--- USER ---
Generate a realistic 24-hour activity schedule for the following person.

## Examples
### Example 1
Person:
- Age: >64
- Sex: male
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 29499-40228
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 5.76-9.6

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}, {"activity": "other", "start": "09:35"}, {"activity": "shop", "start": "10:30"}, {"activity": "home", "start": "12:10"}]}

### Example 2
Person:
- Age: >64
- Sex: female
- Household number of vehicles: 2
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): ≤20148
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: ≤3.2

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}, {"activity": "shop", "start": "14:35"}, {"activity": "home", "start": "15:40"}]}

### Example 3
Person:
- Age: >64
- Sex: male
- Household number of vehicles: 0
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 9.6-17.2968

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}, {"activity": "other", "start": "13:30"}, {"activity": "home", "start": "14:30"}]}

## Person Attributes
- Age: >64
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 5.76-9.6

## Hard Constraints
1. The schedule MUST start with activity "home" at exactly "00:00".
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
4. Times must be strictly increasing throughout the schedule.
5. The final activity must start before 23:59 and will be assumed to end at 24:00.

## Output Format
Respond with a single JSON object in exactly this structure. No other text.

{
  "schedule": [
    {"activity": "home", "start": "00:00"},
    {"activity": "work", "start": "08:30"},
    {"activity": "home", "start": "17:35"}
  ]
}



========================================================================
CONFIG: cot_zero_shot.yaml
========================================================================
--- SYSTEM ---
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
First reason briefly about the person's likely daily pattern, then produce the final JSON schedule.
--- USER ---
Generate a realistic 24-hour activity schedule for the following person.

## Person Attributes
- Age: >64
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 5.76-9.6

## Hard Constraints
1. The schedule MUST start with activity "home" at exactly "00:00".
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
4. Times must be strictly increasing throughout the schedule.
5. The final activity must start before 23:59 and will be assumed to end at 24:00.

## Instructions
First write a brief <reasoning> block: consider whether the person works, their likely wake time, and any other activities they might do. Then produce the schedule JSON inside a <schedule> block.

<reasoning>
[Your step-by-step reasoning here]
</reasoning>

<schedule>
{"schedule": [
  {"activity": "home", "start": 00:00},
  {"activity": "work", "start": 08:30},
  {"activity": "home", "start": 17:35}
]}
</schedule>


========================================================================
CONFIG: cot_few_shot.yaml
========================================================================
--- SYSTEM ---
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
First reason briefly about the person's likely daily pattern, then produce the final JSON schedule.
--- USER ---
Generate a realistic 24-hour activity schedule for the following person.

## Examples
### Example 1
Person:
- Age: >64
- Sex: female
- Household number of vehicles: 2
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): ≤20148
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: ≤3.2

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}, {"activity": "shop", "start": "14:35"}, {"activity": "home", "start": "15:40"}]}

### Example 2
Person:
- Age: >64
- Sex: male
- Household number of vehicles: 0
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 29499-40228
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: unknown

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}]}

### Example 3
Person:
- Age: >64
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: ≤3.2

Schedule:
{"schedule": [{"activity": "home", "start": "00:00"}, {"activity": "other", "start": "07:00"}, {"activity": "home", "start": "09:30"}]}

## Person Attributes
- Age: >64
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 5.76-9.6

## Hard Constraints
1. The schedule MUST start with activity "home" at exactly "00:00".
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
4. Times must be strictly increasing throughout the schedule.
5. The final activity must start before 23:59 and will be assumed to end at 24:00.

## Instructions
First write a brief <reasoning> block: consider whether the person works, their likely wake time, and any other activities they might do. Then produce the schedule JSON inside a <schedule> block.

<reasoning>
[Your step-by-step reasoning here]
</reasoning>

<schedule>
{"schedule": [
  {"activity": "home", "start": 00:00},
  {"activity": "work", "start": 08:30},
  {"activity": "home", "start": 17:35}
]}
</schedule>


========================================================================
CONFIG: high_temp.yaml
========================================================================
--- SYSTEM ---
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
You must respond with valid JSON only — no prose, no explanation, no markdown fences.
--- USER ---
Generate a realistic 24-hour activity schedule for the following person.

## Person Attributes
- Age: >64
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 5.76-9.6

## Hard Constraints
1. The schedule MUST start with activity "home" at exactly "00:00".
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
4. Times must be strictly increasing throughout the schedule.
5. The final activity must start before 23:59 and will be assumed to end at 24:00.

## Output Format
Respond with a single JSON object in exactly this structure. No other text.

{
  "schedule": [
    {"activity": "home", "start": "00:00"},
    {"activity": "work", "start": "08:30"},
    {"activity": "home", "start": "17:35"}
  ]
}



========================================================================
CONFIG: low_temp.yaml
========================================================================
--- SYSTEM ---
You are an expert at generating realistic 24-hour human activity schedules based on person attributes.
You are particulalry skilled at representing the real diversity of human daily patterns, including various work schedules, caregiving responsibilities, and social activities.
You must respond with valid JSON only — no prose, no explanation, no markdown fences.
--- USER ---
Generate a realistic 24-hour activity schedule for the following person.

## Person Attributes
- Age: >64
- Sex: female
- Household number of vehicles: 1
- Has driving licence: yes
- Employment status: retired
- Annual household income (euros): 20148-29499
- Household location type: urban
- Day of survey: monday
- Access/egress distance (km) to public transit: 5.76-9.6

## Hard Constraints
1. The schedule MUST start with activity "home" at exactly "00:00".
2. Allowed activity types (use these exact strings): home, work, education, shop, medical, other, escort, visit
3. Times use 24-hour HH:MM format. Hours range 00–23, minutes range 00–59.
4. Times must be strictly increasing throughout the schedule.
5. The final activity must start before 23:59 and will be assumed to end at 24:00.

## Output Format
Respond with a single JSON object in exactly this structure. No other text.

{
  "schedule": [
    {"activity": "home", "start": "00:00"},
    {"activity": "work", "start": "08:30"},
    {"activity": "home", "start": "17:35"}
  ]
}


