import assert from "node:assert/strict";

import {
  appendNoProxy,
  normalizeProxyServer,
  proxyEnvironment,
} from "../hooks/run_hook.mjs";


assert.equal(normalizeProxyServer("127.0.0.1:7890"), "http://127.0.0.1:7890");
assert.equal(
  normalizeProxyServer("http=127.0.0.1:8080;https=127.0.0.1:8443", "https"),
  "http://127.0.0.1:8443",
);
assert.equal(
  normalizeProxyServer("http=http://proxy.example:8080", "https"),
  "http://proxy.example:8080",
);
assert.equal(normalizeProxyServer(""), null);
assert.equal(appendNoProxy("example.com"), "example.com,127.0.0.1,localhost,::1");

const explicit = proxyEnvironment({
  HTTPS_PROXY: "http://proxy.example:8443",
  HTTP_PROXY: "http://proxy.example:8080",
  NO_PROXY: "example.com",
});
assert.equal(explicit.HTTPS_PROXY, "http://proxy.example:8443");
assert.equal(explicit.HTTP_PROXY, "http://proxy.example:8080");
assert.equal(explicit.NO_PROXY, "example.com,127.0.0.1,localhost,::1");
assert.equal(explicit.NODE_USE_ENV_PROXY, "1");

process.stdout.write("TMCRA hook proxy runner tests passed.\n");
