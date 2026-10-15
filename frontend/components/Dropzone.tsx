"use client";

import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
  maxMb: number;
}

export default function Dropzone({ onFile, disabled, maxMb }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const pick = (files: FileList | null) => {
    const f = files?.[0];
    if (f) onFile(f);
    if (inputRef.current) inputRef.current.value = ""; // allow re-selecting the same file
  };

  return (
    <div
      className={`dropzone ${over ? "over" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) pick(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        id="file-input"
        type="file"
        accept="image/png,image/jpeg"
        disabled={disabled}
        onChange={(e) => pick(e.target.files)}
        className="sr-only"
      />
      <label htmlFor="file-input" className="dz-label">
        <span className="dz-icon" aria-hidden>₹</span>
        <span><strong>Choose an image</strong> or drag and drop</span>
        <span className="muted small">JPG, JPEG or PNG · up to {maxMb} MB · one note per image</span>
      </label>
    </div>
  );
}
