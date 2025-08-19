FROM docker.io/langchain/langgraph-api:3.11
# -- Adding local package . --
ADD src /deps/est_alan
# -- End of local package . --

# -- Installing all local dependencies --

RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e /deps/*

# -- End of local dependencies install --


# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license &&     touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing pip from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel &&     rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* &&     find /usr/local/bin -name "pip*" -delete || true
# pip removal for wolfi
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* &&     find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel
# -- End of pip removal --

# -- Copy Local files
COPY entrypoint.sh /entrypoint.sh
COPY graph.json /graph.json
COPY est-alan-dev-account.json /deps/est_alan/est-alan-dev-account.json


RUN chmod +x /entrypoint.sh

WORKDIR /deps/est_alan

ENTRYPOINT ["/entrypoint.sh"]

