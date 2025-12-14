"use client";

import { useState, useEffect } from "react";

export default function Debug() {
  const [images, setImages] = useState<string[]>([]);

  useEffect(() => {
    const fetchImages = async () => {
      try {
        const response = await fetch("/api/debug-images");
        const data = await response.json();
        setImages(data.images || []);
      } catch (error) {
        console.error("Failed to fetch:", error);
      }
    };

    fetchImages();
    const interval = setInterval(fetchImages, 3000); // Every 3 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4 bg-black text-white min-h-screen">
      <h1 className="text-xl mb-4">Live Training ({images.length} images)</h1>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {images.slice(0, 6).map((image) => (
          <div key={image} className="border border-gray-600 rounded p-2">
            <img
              src={`/debug/${image}`}
              alt={image}
              className="w-full h-32 object-cover rounded"
            />
            <p className="text-xs mt-1">{image}</p>
          </div>
        ))}
      </div>

      {images.length === 0 && <p className="text-gray-400">No images yet...</p>}
    </div>
  );
}
