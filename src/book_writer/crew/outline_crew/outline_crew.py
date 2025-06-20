from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
import os 

llm = LLM( model="gpt-4o-mini-2024-07-18",  
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1")

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