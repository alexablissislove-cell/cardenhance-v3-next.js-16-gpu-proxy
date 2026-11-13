"use client";

import Image from "next/image";
import { useState } from "react";

interface ComparisonViewProps {
  sourceUrl: string;
  resultUrl: string | null;
  sourceName: string;
}

export function ComparisonView({ sourceUrl, resultUrl, sourceName }: ComparisonViewProps) {
  const [position, setPosition] = useState(50);
  const hasResult = resultUrl !== null;

  return (
    <div className="comparison-frame" data-comparing={hasResult || undefined}>
      <Image
        className="comparison-image"
        src={sourceUrl}
        alt={`Original ${sourceName}`}
        fill
        unoptimized
        priority
        draggable={false}
        sizes="(max-width: 1024px) 100vw, 70vw"
      />
      {hasResult && (
        <>
          <div
            className="comparison-after"
            style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
          >
            <Image
              className="comparison-image"
              src={resultUrl}
              alt={`Restored ${sourceName}`}
              fill
              unoptimized
              draggable={false}
              sizes="(max-width: 1024px) 100vw, 70vw"
            />
          </div>
          <div className="comparison-line" style={{ left: `${position}%` }} aria-hidden="true">
            <span className="comparison-grip">‹ ›</span>
          </div>
          <span className="comparison-label comparison-label-before">Before</span>
          <span className="comparison-label comparison-label-after">After</span>
          <input
            className="comparison-range"
            type="range"
            min="0"
            max="100"
            value={position}
            onChange={(event) => setPosition(Number(event.target.value))}
            aria-label="Reveal restored image"
            aria-valuetext={`${position}% restored image visible`}
          />
        </>
      )}
      {!hasResult && <span className="comparison-label comparison-label-source">Source</span>}
    </div>
  );
}
