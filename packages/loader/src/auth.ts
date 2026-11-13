/**
 * An access token for gated and private models on the Hugging Face Hub. One per thread (a client
 * hands its workers the token with each request). It is sent as `Authorization: Bearer …`, and only
 * to https://huggingface.co itself: never to another endpoint (a mirror, a server given as
 * `endpoint`, the URL of one .gguf file), where it would be handed to whoever runs that server.
 * The Hub answers file requests with a redirect to its storage; browsers drop the header on a
 * redirect to another origin, so the token stops at the Hub.
 */
let token: string | undefined;

/** Hosts the token is for. */
const TOKEN_HOSTS = new Set(["huggingface.co"]);

/**
 * Sets the token later requests to the Hub carry (undefined or null: none). Create a read-only,
 * fine-grained token for this: any script on the page can read what the page holds.
 */
export function setAccessToken(value?: string | null): void {
  if (value === undefined || value === null || value === "") {
    token = undefined;
    return;
  }
  // A header value: one line of visible ASCII. Anything else is a mistake (a pasted line break) or an attempt to add headers.
  if (typeof value !== "string" || value.length > 512 || !/^[\x21-\x7e]+$/.test(value)) throw new Error("setAccessToken: the token must be one line of printable characters without spaces");
  token = value;
}

/** The token set with `setAccessToken`, for handing to a worker. */
export function accessToken(): string | undefined {
  return token;
}

/** True when a request to `url` carries the token: one is set, and the URL is the Hub's, over https. */
export function sendsToken(url: string): boolean {
  if (!token) return false;
  try {
    const u = new URL(url);
    return u.protocol === "https:" && TOKEN_HOSTS.has(u.hostname) && (u.port === "" || u.port === "443");
  } catch {
    return false;
  }
}

/** `fetch` for repo files: adds the token where `sendsToken` allows it, and nothing anywhere else. */
export function hubFetch(url: string, init: { headers?: Record<string, string> } = {}): Promise<Response> {
  if (!sendsToken(url)) return fetch(url, init);
  return fetch(url, { ...init, headers: { ...init.headers, Authorization: `Bearer ${token!}` } });
}

/** "<url>: HTTP <status>", with what to do about a refusal by the Hub. */
export function httpError(url: string, status: number): string {
  let hint = "";
  if (status === 401 || status === 403) {
    let hub = false;
    try {
      hub = TOKEN_HOSTS.has(new URL(url).hostname);
    } catch {
      // Not a URL: no hint.
    }
    if (hub) {
      hint = sendsToken(url)
        ? " (the access token does not open this model: accept the model's terms on its Hub page, and check the token can read it)"
        : " (a gated or private model: set an access token with setAccessToken, after accepting the model's terms on its Hub page)";
    } else if (status === 401) hint = " (gated or private model?)";
  }
  return `${url}: HTTP ${status}${hint}`;
}
