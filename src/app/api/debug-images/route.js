import fs from "fs";
import { NextResponse } from "next/server";
import path from "path";

export async function GET() {
  try {
    const debugDir = path.join(process.cwd(), "public", "debug");

    if (!fs.existsSync(debugDir)) {
      return NextResponse.json({ images: [] });
    }

    const files = fs
      .readdirSync(debugDir)
      .filter((file) => file.endsWith(".jpg"))
      .sort((a, b) => {
        const stepA = parseInt(a.match(/step_(\d+)/)?.[1] || "0");
        const stepB = parseInt(b.match(/step_(\d+)/)?.[1] || "0");
        return stepB - stepA; // newest first
      });

    return NextResponse.json({ images: files });
  } catch (error) {
    return NextResponse.json({ images: [] });
  }
}
