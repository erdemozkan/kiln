/**
 * Keeps each GPU submit short. A submit is one command buffer, and the GPU only switches to other
 * work, such as drawing the screen, between command buffers; on a slow integrated GPU one 128-token
 * prefill chunk can take most of a second, which freezes the display and can trip the OS's GPU hang
 * watchdog. So work is cut into pieces of at most a budget of estimated GPU time.
 *
 * The estimate is a cost per unit (a dispatch, or a dispatch-row for prefill) learned from measured
 * times. Until a measurement arrives it assumes a slow GPU (the default cost is an Intel Iris Plus
 * 650's), so the first submits err towards too many pieces rather than too few.
 */
export class SubmitPacer {
  private msPerUnit: number;

  constructor(initialMsPerUnit: number) {
    this.msPerUnit = initialMsPerUnit;
  }

  /** Pieces to cut `dispatches` dispatches of `units` total cost into (1 = one submit). */
  pieces(units: number, dispatches: number, budgetMs: number): number {
    return Math.max(1, Math.min(dispatches, Math.ceil((units * this.msPerUnit) / budgetMs)));
  }

  /** Records that work of `units` cost took `ms` on the GPU. */
  observe(ms: number, units: number): void {
    if (!(ms > 0) || !(units > 0)) return;
    const v = ms / units;
    // Quick to learn the GPU is slower than thought, slower to believe it is faster.
    this.msPerUnit += (v > this.msPerUnit ? 0.5 : 0.2) * (v - this.msPerUnit);
  }

  /** Cuts a dispatch list into `pieces` runs of nearly equal length, in order. */
  static split<T>(list: readonly T[], pieces: number): T[][] {
    const out: T[][] = [];
    for (let i = 0; i < pieces; i++) out.push(list.slice(Math.floor((i * list.length) / pieces), Math.floor(((i + 1) * list.length) / pieces)));
    return out.filter((p) => p.length);
  }
}
