import neptune.new as neptune

class NeptuneLogger():
    def __init__(self) -> None:
        #Neptune initialization
        self.run = neptune.init(
            project="annabadalyan/NLP-discursive-repertoirs",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkOGMyZmE4NS04ZDkwLTQ5YzItODhiNC03NmJiMTQzMDNiYmMifQ==",
        )
    