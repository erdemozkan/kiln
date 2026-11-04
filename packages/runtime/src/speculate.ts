/**
 * Prompt-lookup drafting for speculative decoding: when the last few tokens appeared earlier in the
 * prompt or the output, guess that what followed them then comes next. No draft model; it pays off
 * when the output repeats its input (summaries, code edits, JSON echoing given values).
 *
 * The most recent earlier occurrence of the longest matching suffix (`maxNgram` tokens down to
 * `minNgram`) wins. Returns up to `maxDraft` tokens, or none.
 */
export function lookupDraft(tokens: readonly number[], maxDraft: number, maxNgram = 3, minNgram = 2): number[] {
  const L = tokens.length;
  if (maxDraft <= 0) return [];
  for (let n = Math.min(maxNgram, L - 1); n >= minNgram; n--) {
    // Candidate starts i where tokens[i..i+n) equals the suffix, latest first; i + n < L keeps at
    // least one following token and excludes the suffix itself.
    for (let i = L - n - 1; i >= 0; i--) {
      let j = 0;
      while (j < n && tokens[i + j] === tokens[L - n + j]) j++;
      if (j === n) return tokens.slice(i + n, Math.min(i + n + maxDraft, L));
    }
  }
  return [];
}

/**
 * How many draft tokens a verify pass confirmed. Its inputs were the last token followed by `draft`;
 * `sampled[i]` is the token the model chose after input i. Draft token i stands while every earlier one
 * did and the model chose it too.
 */
export function acceptedLength(draft: readonly number[], sampled: ArrayLike<number>): number {
  let n = 0;
  while (n < draft.length && sampled[n] === draft[n]) n++;
  return n;
}

/**
 * Decides, per draft, whether checking it is worth a verify pass on this GPU. A verify pass costs more
 * than a plain decode step (how much depends on the GPU), so it has to keep enough draft tokens to pay
 * for itself. Both costs and the share of draft tokens kept are learned as decoding goes. While drafts
 * do not pay, decoding stays plain, and each skipped draft wins back a little optimism, so a later
 * stretch of repetitive text is still picked up.
 */
export class DraftGate {
  private passMs: number | undefined;
  private tokenMs: number | undefined;
  /** Share of draft tokens kept, smoothed; starts optimistic. */
  private keep = 1;

  worth(draftLength: number): boolean {
    // Plain steps are measured first, then one pass, so the comparison has both sides.
    if (this.tokenMs === undefined) return false;
    if (this.passMs === undefined) return true;
    // A pass yields the model's own token plus the kept draft; plain decoding makes passMs / tokenMs.
    return 1 + this.keep * draftLength > this.passMs / this.tokenMs;
  }

  verified(ms: number, drafted: number, kept: number): void {
    this.passMs = ema(this.passMs, ms);
    this.keep = ema(this.keep, kept / drafted);
  }

  plain(ms: number, tokens: number, skippedDraft: boolean): void {
    this.tokenMs = ema(this.tokenMs, ms / tokens);
    if (skippedDraft) this.keep += (1 - this.keep) * 0.02;
  }
}

const ema = (old: number | undefined, v: number): number => (old === undefined ? v : old + 0.3 * (v - old));
