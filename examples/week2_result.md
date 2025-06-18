# AGI Agent Builder: Week 2 Result

## System Overview
A modular multi-agent system that can:
- Read and extract content from documents (PDF/text)
- Plan and decompose tasks into actionable steps
- Use specialized tools to perform subtasks (summarization, keyword extraction, calculation)
- Chain agents for end-to-end task execution

## Example Run

**Input Document:** `ComputerScienceResumedocx.pdf`

**User Task:**
```
Summarize this resume and highlight key skills. Also, calculate the mean of [85, 90, 78, 92].
```

---

**Planner Agent Output (Structured Steps):**
```json
{
  "task": "Summarize this resume and highlight key skills. Also, calculate the mean of [85, 90, 78, 92].",
  "steps": [
    "Write a short summary of your experience and skills. Highlight the most important skills and accomplishments.",
    "Extract the top 5 keywords or skills from the resume.",
    "Calculate the mean of [85, 90, 78, 92]."
  ]
}
```

---

**Tool Agent Output:**

- **Step 1:**
  - *Summary Output*: (Concise, professional summary of the resume)

- **Step 2:**
  - *Keywords Output*: (List of top 5 skills/keywords)

- **Step 3:**
  - *Calculation Output*: `The result of the calculation is: 86.25`

---

## System Chain
```
ReaderAgent → PlannerAgent → ToolAgent → Output
```

## Notes
- All model calls are local (no API keys required)
- Tools are modular and easily extendable
- Output is structured for easy downstream use

---
*This is a sample result for Week 2 of the AGI Agent Builder Plan. The system is ready for further expansion in Week 3 (Critique & Refinement).*
