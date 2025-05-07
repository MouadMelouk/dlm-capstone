import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { v4 as uuidv4 } from 'uuid';

/**
 * POST /api/uploadFile
 * Expects FormData with a single "file" field for a video.
 * 1) Saves the file to a temp folder on the server.
 * 2) Spawns upload_file.py, passing the local temp path & HPC destination path.
 * 3) Returns { hpcPath: string } if successful.
 */

export async function POST(req: Request) {
  try {
    // 1. Parse the incoming form data
    const formData = await req.formData();
    const file = formData.get('file') as Blob | null;

    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    // 2. Generate a unique ID and figure out the local temp path + HPC filename
    const uniqueId = uuidv4().slice(0, 8); // for the HPC file name
    const originalFilename = (file as File).name ?? 'uploaded-video';
    const ext = path.extname(originalFilename) || '.mp4'; // fallback .mp4 if no extension
    const tempLocalFilePath = path.join(os.tmpdir(), uniqueId + ext);
    
    // HPC path you want to store the file at, e.g. /scratch/mmm9912/Capstone/FRONT_END_STORAGE/videos/{uniqueId}.mp4
    // We'll pass just the subfolder + filename to the Python script if that's how you'd prefer. 
    // Or you can pass the full absolute path. 
    const hpcRelativePath = `videos/${uniqueId}${ext}`; // e.g. "videos/1234-uuid.mp4"

    // 3. Write the uploaded file to a temp path
    const arrayBuffer = await file.arrayBuffer();
    fs.writeFileSync(tempLocalFilePath, Buffer.from(arrayBuffer));

    // 4. Spawn (run) the python script with arguments: local temp path, HPC sub-path
    //    Adjust the path to upload_file.py as needed. Right now we assume it's in the *root* 
    //    of your Next.js project, i.e., `capstone_frontend/../upload_file.py`.
    // REPLACE THIS WITH THE SCRIPT PATH
    const pythonScriptPath = "/Users/Apple/Documents/Capstone_Front_End/capstone_frontend/upload_file.py";

    // We'll collect stdout and stderr
    let stdoutData = '';
    let stderrData = '';

    // Return a promise so we can `await` the python process finishing
    // REPLACE THIS WITH THE VENV LOCATION
    const result = await new Promise<string>((resolve, reject) => {
      const pyProcess = spawn('/Users/Apple/Documents/Capstone_Front_End/capstone_frontend_venv/bin/python', [
        pythonScriptPath,
        tempLocalFilePath,  // arg1
        hpcRelativePath     // arg2
      ]);

      pyProcess.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });

      pyProcess.stderr.on('data', (data) => {
        stderrData += data.toString();
      });

      pyProcess.on('close', (code) => {
        if (code === 0) {
          // We expect the python script to print out the final HPC path on success.
          // E.g., /scratch/mmm9912/Capstone/FRONT_END_STORAGE/videos/abcd-1234.mp4
          // We can parse it from stdoutData. For simplicity, let's assume the script prints 
          // ONLY the HPC path as the last line or so. You can refine as needed.
          const lines = stdoutData.trim().split('\n');
          const lastLine = lines[lines.length - 1];
          resolve(lastLine); 
        } else {
          reject(`upload_file.py exited with code ${code}.\nStderr: ${stderrData}\nStdout: ${stdoutData}`);
        }
      });
    });

    // 5. Clean up the temp file (optional)
    fs.unlinkSync(tempLocalFilePath);

    // 6. Return HPC path to the client
    // `result` is the HPC path we parsed from the python script
    return NextResponse.json({ hpcPath: result }, { status: 200 });

  } catch (err) {
    console.error('Error in uploadFile route:', err);
    return NextResponse.json({ error: 'File upload failed' }, { status: 500 });
  }
}
