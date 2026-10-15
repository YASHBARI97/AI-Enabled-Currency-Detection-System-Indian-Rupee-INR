// Client-side pre-checks for fast feedback. The server remains the source of truth.
export const ALLOWED_TYPES = ["image/jpeg", "image/png"];
export const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"];

export interface FileCheck {
  ok: boolean;
  message?: string;
}

export function checkFile(file: { name: string; type: string; size: number }, maxMb: number): FileCheck {
  if (file.size === 0) return { ok: false, message: "The selected file is empty." };
  const lower = file.name.toLowerCase();
  const extOk = ALLOWED_EXTENSIONS.some((e) => lower.endsWith(e));
  if (!extOk || (file.type && !ALLOWED_TYPES.includes(file.type))) {
    return { ok: false, message: "Only JPG, JPEG or PNG images are supported." };
  }
  if (file.size > maxMb * 1024 * 1024) {
    return { ok: false, message: `File exceeds the ${maxMb} MB limit.` };
  }
  return { ok: true };
}

export function formatPercent(x: number | null | undefined): string {
  return x == null ? "–" : `${(x * 100).toFixed(1)}%`;
}

export function prettyLabel(label: string | null | undefined): string {
  if (!label) return "–";
  return label.replace(/^INR_/, "₹");
}
