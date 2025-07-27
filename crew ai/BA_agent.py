import os
from crewai import Agent, Task, Crew

api_key = os.environ["MISTRAL_API_KEY"]

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://api.mistral.ai/v1"
os.environ["OPENAI_MODEL_NAME"] = "mistral-small"

# 1. Agents
# 1.1 Requirements Planner
requirements_planner = Agent(
    role="Requirements Planner",
    goal="Plan and prioritize the necessary documents for the project: {project_description}",
    backstory="You are responsible for breaking down the project description into clear, actionable requirements. Your work will provide the foundation for the BA document creation.",
    allow_delegation=False,
    verbose=True,
)

# 1.2 Document Writer
document_writer = Agent(
    role="Document Writer",
    goal="Write comprehensive BA documents for the project: {project_description}",
    backstory="Using the requirements provided by the Requirements Planner, you create detailed documents such as the BRD, FRS, and use cases. Your work aligns with industry standards and the specific needs of the project.",
    allow_delegation=False,
    verbose=True,
)

# 1.3 Quality Checker
quality_checker = Agent(
    role="Quality Checker",
    goal="Review and validate the BA documents to ensure they meet quality and accuracy standards.",
    backstory="You are tasked with reviewing the generated documents to ensure clarity, completeness, and adherence to the project's objectives and requirements.",
    allow_delegation=False,
    verbose=True,
)

# 2. Tasks
# 2.1 Plan Requirements
plan_requirements = Task(
    description=(
        "1. Analyze the project description and identify high-level business objectives.\n"
        "2. Outline the key documents required (e.g., BRD, FRS, Use Cases).\n"
        "3. Create a prioritized list of requirements for document creation."
    ),
    expected_output="A detailed requirements plan outlining necessary BA documents and key points for each.",
    agent=requirements_planner,
)

# 2.2 Write Documents
write_documents = Task(
    description=(
        "1. Use the requirements plan to draft the BRD, FRS, and use case documents.\n"
        "2. Ensure each document is detailed, structured, and aligned with project goals.\n"
        "3. Include relevant diagrams, workflows, and specifications as needed."
    ),
    expected_output="A complete set of BA documents in markdown format, including BRD, FRS, and use cases.",
    agent=document_writer,
)

# 2.3 Validate Documents
validate_documents = Task(
    description=(
        "1. Review the generated documents for clarity, accuracy, and completeness.\n"
        "2. Ensure all requirements are addressed and no critical information is missing.\n"
        "3. Provide feedback for improvement or finalize the documents."
    ),
    expected_output="Validated and finalized BA documents ready for stakeholder review.",
    agent=quality_checker,
)

crew = Crew(
    agents=[requirements_planner, document_writer, quality_checker],
    tasks=[plan_requirements, write_documents, validate_documents],
    verbose=True,
)

results = crew.kickoff(
    inputs={
        "project_description": "Develop a CRM application for managing customer interactions and automating sales pipelines."
    }
)

print(type(results))
# print(results)
for result in results:
    print("resulttt", result)

# saving
file_name = "BA_Docs.md"
with open(file_name, "w") as file:
    file.write(str(results))
print(f"Document saved as {file_name}")

# Save Documents to Files
# if "Validated and finalized BA documents ready for stakeholder review." in results:
# for doc_type, content in results.items():
#     if isinstance(content, str):
#         file_name = f"{doc_type.replace(' ', '_').lower()}.md"
#         with open(file_name, "w") as file:
#             file.write(content)
#         print(f"Document saved as {file_name}")
