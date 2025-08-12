def create_initialization_node(init_func=None, private_state_key=None, **kwargs):
    def initialization_node(state):
        if private_state_key is not None:
            state = state.get(private_state_key, {})
        
        initialization = state.get("initialization", False)

        if not initialization:
            updated_state = dict(kwargs)
            if init_func and callable(init_func):
                state2 = init_func(updated_state)
                for key in state2.keys():
                    updated_state[key] = state2[key]

            updated_state["initialization"] = True
        else:
            updated_state = {}

        if private_state_key is not None:
            updated_state = {private_state_key: updated_state}

        return updated_state

    return initialization_node
