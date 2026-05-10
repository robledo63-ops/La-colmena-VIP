import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq

@CrewBase
class LaColmenaCrew():
    """Crew de Inmobiliaria Bajakey"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def llm(self):
        return ChatGroq(model_name="llama3-70b-8192", groq_api_key=os.environ.get("GROQ_API_KEY"))

    @agent
    def consultor(self) -> Agent:
        return Agent(config=self.agents_config, llm=self.llm(), verbose=True)

    @task
    def tarea_guia_whatsapp(self) -> Task:
        return Task(config=self.tasks_config)

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
