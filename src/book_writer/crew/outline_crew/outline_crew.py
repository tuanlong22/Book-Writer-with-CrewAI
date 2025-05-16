from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel

llm = LLM(
    provider="ollama",
    model="ollama/llama3.2",
    api_base="http://localhost:11434",
    config={
        "temperature": 0.7,
        "seed": 42
    }
)

class Outline(BaseModel):
    """Outline of the book"""
    total_chapters: int
    titles: list[str]

@CrewBase
class OutlineCrew:
    """Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_agent(self) -> Agent:
        return Agent(config=self.agents_config["research_agent"],
                     llm=llm)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])
    
    @agent
    def outline_writer(self) -> Agent:
        return Agent(config=self.agents_config["outline_writer"],
                     llm=llm)

    @task
    def write_outline(self) -> Task:
        return Task(config=self.tasks_config["write_outline"],
                    output_pydantic=Outline)

    @crew
    def crew(self) -> Crew:
        """Creates the Outline Crew"""

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True)