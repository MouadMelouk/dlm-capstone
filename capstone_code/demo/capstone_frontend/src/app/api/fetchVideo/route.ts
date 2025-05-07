import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { promisify } from 'util';

// We'll read the entire file into memory in this naive example
const readFileAsync = promisify(fs.readFile);

export async function GET(req: Request) {
  try {
    // 1) Parse query param
    // The full HPC path might be passed as ?path=..., e.g. /scratch/.../videos/abcd-1234.mp4
    const { searchParams } = new URL(req.url);
    const hpcPath = searchParams.get('path');
    if (!hpcPath) {
      return NextResponse.json({ error: 'Missing ?path=' }, { status: 400 });
    }

    // 2) We define a local temp file to download into
    const fileName = path.basename(hpcPath); // e.g. abcd-1234.mp4
    const localTempFile = path.join(os.tmpdir(), fileName);

    // 3) Spawn the fetch_video.py script to scp GET the HPC file
    console.log('Fetching HPC video:', hpcPath, '=>', localTempFile);

    // REPLACE THIS WITH THE SCRIPT LOCATION
    const pythonScriptPath = "/Users/Apple/Documents/Capstone_Front_End/capstone_frontend/fetch_video.py"; 
    // or adjust if your file is in a different location:
    // const pythonScriptPath = path.join(process.cwd(), 'fetch_video.py');

    let stdoutData = '';
    let stderrData = '';

    // REPLACE THIS WITH THE VENV LOCATION
    const exitCode = await new Promise<number>((resolve, reject) => {
      const py = spawn('/Users/Apple/Documents/Capstone_Front_End/capstone_frontend_venv/bin/python', [
        pythonScriptPath,
        hpcPath,         // arg1 = remote HPC path
        localTempFile    // arg2 = local file path
      ]);

      py.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });
      py.stderr.on('data', (data) => {
        stderrData += data.toString();
      });
      py.on('close', (code) => resolve(code ?? 1));
      py.on('error', (err) => reject(err));
    });

    if (exitCode !== 0) {
      console.error('fetch_video.py error:', stderrData, stdoutData);
      return NextResponse.json({ error: 'Failed to fetch video from HPC.' }, { status: 500 });
    }

    // 4) The Python script succeeded, so the file is presumably at localTempFile
    //    We'll read it into a Buffer for a naive response
    const fileBuffer = await readFileAsync(localTempFile);

    // 5) Optionally remove local temp file to keep the system clean
    fs.unlink(localTempFile, () => null);

    // 6) Return it as "video/mp4" or guess the content-type from extension
    //    This naive approach just returns the entire file in one shot
    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'video/mp4', 
        // Potentially handle .mov, .avi etc. if you want 
        'Content-Length': fileBuffer.length.toString(),
      },
    });
  } catch (err) {
    console.error('Error in fetchVideo route:', err);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
