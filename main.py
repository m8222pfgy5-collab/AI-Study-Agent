
import json
from pathlib import Path
from datetime import date
from agents import Agent, Runner, function_tool

FILE = Path(__file__).resolve().parent / "tasks.json"


# Read saved tasks
def load_tasks():
    if not FILE.exists():
        return []

    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert old Version 2 tasks to Version 3
    for i, task in enumerate(data):
        if isinstance(task, str):
            data[i] = {
                "id": i + 1,
                "name": task,
                "completed": False,
                "due_date": None
            }

    return data


def save_tasks():
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=4)


tasks = load_tasks()


# Find a task by ID
def find_task(task_id):
    for task in tasks:
        if task["id"] == task_id:
            return task
    return None


# Tool 1: Add task
@function_tool
def add_task(name: str) -> str:
    """Add a new study task."""
    new_id = max(
        (task["id"] for task in tasks),
        default=0
    ) + 1

    tasks.append({
        "id": new_id,
        "name": name,
        "completed": False,
        "due_date": None
    })

    save_tasks()
    return f"Added task {new_id}: {name}"


# Tool 2: List tasks
@function_tool
def list_tasks() -> str:
    """List all tasks with IDs, status and due dates."""
    if not tasks:
        return "No tasks found."

    lines = []

    for task in tasks:
        status = "Done" if task["completed"] else "Pending"
        due = task["due_date"] or "No deadline"

        lines.append(
            f'ID {task["id"]}: {task["name"]} '
            f'| {status} | Due: {due}'
        )

    return "\n".join(lines)


# Tool 3: Complete task
@function_tool
def complete_task(task_id: int) -> str:
    """Mark a task completed using its ID."""
    task = find_task(task_id)

    if task is None:
        return "Task not found."

    task["completed"] = True
    save_tasks()

    return f'Completed: {task["name"]}'


# Tool 4: Delete task
@function_tool
def delete_task(task_id: int) -> str:
    """Delete a task using its ID."""
    task = find_task(task_id)

    if task is None:
        return "Task not found."

    tasks.remove(task)
    save_tasks()

    return f'Deleted: {task["name"]}'


# Tool 5: Set deadline
@function_tool
def set_deadline(task_id: int, due_date: str) -> str:
    """Set a deadline in YYYY-MM-DD format."""
    task = find_task(task_id)

    if task is None:
        return "Task not found."

    try:
        date.fromisoformat(due_date)
    except ValueError:
        return "Invalid date. Use YYYY-MM-DD."

    task["due_date"] = due_date
    save_tasks()

    return f'Deadline updated: {task["name"]} -> {due_date}'


# Create AI Agent
agent = Agent(
    name="Study Agent",
    instructions=(
        "You are an AI study assistant. "
        "Help students manage study tasks. "
        "Always reply in Chinese. "
        "Use list_tasks to identify task IDs before "
        "completing, deleting or updating tasks. "
        "If the task is ambiguous, ask the user to clarify. "
        "Before deleting a task, ask for confirmation. "
        "Only delete after explicit confirmation. "
        "Use YYYY-MM-DD for deadlines. "
        "Today's date is "
        f"{date.today().isoformat()}. "
        "Never invent a task or deadline."
    ),
    tools=[
        add_task,
        list_tasks,
        complete_task,
        delete_task,
        set_deadline
    ],
)

print("AI Study Agent Version 3 started!")
print("Type 'exit' to quit.")

history = []

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    try:
        history.append({
            "role": "user",
            "content": user_input
        })

        result = Runner.run_sync(agent, history)
        print("Agent:", result.final_output)

        history = result.to_input_list()

    except Exception as e:
        history.pop()
        print("Error:", e)
