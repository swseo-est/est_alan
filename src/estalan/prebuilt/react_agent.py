import langgraph.prebuilt


def create_react_agent(*args, **kwargs):
    agent = langgraph.prebuilt.create_react_agent(*args, **kwargs)
    return agent