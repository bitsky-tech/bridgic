from bridgic.core.automa.worker import Worker

class DoubleWorker(Worker):
    async def arun(self, user_input: int):
        return user_input * 2
    
    