export const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;
export const DEFAULT_MAX_IMAGE_BYTES = 4_000_000;

export type OutputFormat = "png" | "jpeg";

export interface ServiceStatus {
  configured: boolean;
  ready: boolean;
  maxBytes: number;
  mode?: "paired" | "unpaired";
  generator?: string;
  device?: string;
  checkpointEpoch?: number;
  detail?: string;
}

export interface ImageDimensions {
  width: number;
  height: number;
}
