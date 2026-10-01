import { useState } from "react";
import type { IdentityAsset } from "./identity-assets";

export default function IdentityImage({
  asset,
}: {
  asset: IdentityAsset | undefined;
}) {
  const [failedSource, setFailedSource] = useState<string>();
  const failed = !asset || failedSource === asset.src;
  return (
    <span
      className="identity-image identity-logo"
      data-unavailable={failed || undefined}
      aria-hidden="true"
    >
      {failed ? (
        <span className="identity-unavailable" title="Team logo unavailable">
          -
        </span>
      ) : (
        <img
          src={asset.src}
          alt=""
          width={asset.width}
          height={asset.height}
          loading="lazy"
          decoding="async"
          onError={() => setFailedSource(asset.src)}
        />
      )}
    </span>
  );
}
