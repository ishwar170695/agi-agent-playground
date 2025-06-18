def results_to_markdown(final_output):
    md = "# AGI Agent Pipeline Results\n"
    for i, item in enumerate(final_output, 1):
        md += f"\n## Step {i}: {item['step']}\n"
        md += f"\n**Main Output:**\n\n{item['output']}\n"
        md += f"\n**Critique:**\n\n{item['critique']}\n"
        md += f"\n**Improved Output:**\n\n{item['improved']}\n"
    return md
