import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { promisify } from 'util';

const readFileAsync = promisify(fs.readFile);

/**
 * GET /api/fetchMedia?path=<fullHPCpath>
 * Example: /api/fetchMedia?path=/scratch/.../someImage.jpg
 */
export async function GET(req: Request) {
  try {
    // 1) Parse the HPC path from query
    const { searchParams } = new URL(req.url);
    const hpcPath = searchParams.get('path');
    if (!hpcPath) {
      return NextResponse.json({ error: 'Missing ?path=' }, { status: 400 });
    }

    // 2) Local temp file to download into
    const filename = path.basename(hpcPath); // e.g. 'someImage.jpg'
    const localTemp = path.join(os.tmpdir(), filename);

    // 3) Spawn fetch_media.py to scp-get from HPC
    // REPLACE THIS WITH THE SCRIPT LOCATION
    const pythonScriptPath = "/Users/Apple/Documents/Capstone_Front_End/capstone_frontend/fetch_media.py";
    // Adjust if your script is located differently, e.g.:
    // const pythonScriptPath = path.join(process.cwd(), 'fetch_media.py');

    let stdoutData = '';
    let stderrData = '';

    const exitCode: number = await new Promise((resolve, reject) => {
      // REPLACE THIS WITH THE VENV LOCATION
      const py = spawn(
        '/Users/Apple/Documents/Capstone_Front_End/capstone_frontend_venv/bin/python', 
        [pythonScriptPath, hpcPath, localTemp]
      );

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
      console.error('fetch_media.py error:', stderrData, stdoutData);
      return NextResponse.json({ error: 'Failed to fetch media from HPC.' }, { status: 500 });
    }

    // 4) The Python script succeeded. The file is local now.
    const fileBuf = await readFileAsync(localTemp);

    // 5) Optionally remove local temp file to keep server clean
    fs.unlink(localTemp, () => null);

    // 6) Guess content-type from extension
    const ext = path.extname(filename).toLowerCase();
    let contentType = 'application/octet-stream';
    if (ext === '.mp4' || ext === '.mov' || ext === '.avi') {
      contentType = 'video/mp4'; // or handle .mov, .avi specifically
    } else if (ext === '.jpg' || ext === '.jpeg') {
      contentType = 'image/jpeg';
    } else if (ext === '.png') {
      contentType = 'image/png';
    } else if (ext === '.gif') {
      contentType = 'image/gif';
    }

    // 7) Return the file
    return new NextResponse(fileBuf, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Length': fileBuf.length.toString(),
      },
    });
  } catch (err) {
    console.error('Error in fetchMedia route:', err);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
