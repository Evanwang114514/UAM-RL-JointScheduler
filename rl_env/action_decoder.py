class ActionDecoder:
    """
    Discrete action -> Scenario action dict
    """

    def __init__(self, from_vertiports, to_vertiport):
        self.from_vertiports = from_vertiports
        self.to_vertiport = to_vertiport

    def decode(self, action: int, pid: str):
        from_vid = self.from_vertiports[action]

        return {
            pid: {
                "mode": "UAM",
                "from_vertiport": from_vid,
                "to_vertiport": self.to_vertiport
            }
        }
