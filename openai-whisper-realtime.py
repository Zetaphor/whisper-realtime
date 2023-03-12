import time
import sounddevice as sd
import numpy as np

import whisper

import asyncio
import queue
import sys

import websockets
import socket
from threading import Thread, Event
from collections import deque

server_host = '127.0.0.1'
server_port = 8081

# SETTINGS
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
MODEL_TYPE="tiny.en"

# pre-set the language to avoid autodetection
LANGUAGE="English"

# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds.
BLOCKSIZE=24678

# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_THRESHOLD=1000

# number of samples in one buffer that are allowed to be higher than threshold
SILENCE_RATIO=2

HAS_DATA_WAITING=False

FIRST_RUN=True

DEBUG=False

latest_data = ""

global_ndarray = None
model = whisper.load_model(MODEL_TYPE)

from collections import deque

print("Starting server...")

if DEBUG:
	print("Silence threshold division", SILENCE_THRESHOLD/15)

async def inputstream_generator():
	"""Generator that yields blocks of input data as NumPy arrays."""
	q_in = asyncio.Queue()
	loop = asyncio.get_event_loop()

	def callback(indata, frame_count, time_info, status):
		loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

	stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
	with stream:
		while True:
			indata, status = await q_in.get()
			yield indata, status


async def process_audio_buffer():
	global global_ndarray
	global HAS_DATA_WAITING
	global latest_data
	async for indata, status in inputstream_generator():

		indata_flattened = abs(indata.flatten())

		if DEBUG:
			print("\nConcatenated buffers:", np.average((indata_flattened[-100:-1])))

		# discard buffers that contain mostly silence
		if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO) and not HAS_DATA_WAITING:
			if DEBUG:
				print("\nFlattened:", np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size, "Queued:", HAS_DATA_WAITING)
			continue

		if (global_ndarray is not None):
			global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
		else:
			global_ndarray = indata

		# concatenate buffers if the end of the current buffer is not silent
		if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/15):
			print("Recording audio, waiting for silence...")
			if DEBUG:
				print("\nStill waiting for silence, concatenating...")
				print("Average:", np.average((indata_flattened[-100:-1])))
			HAS_DATA_WAITING = True
			continue
		else:
			HAS_DATA_WAITING = False
			print("\nTranscribing...")
			start_time = time.perf_counter()
			local_ndarray = global_ndarray.copy()
			global_ndarray = None
			indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
			result = model.transcribe(indata_transformed, language=LANGUAGE)
			print("Output:", result["text"])
			latest_data = result["text"]
			end_time = time.perf_counter()
			print(f"Transcribed in {end_time - start_time:0.4f} seconds\n")

		del local_ndarray
		del indata_flattened



async def main():
	print('\nActivating wire...\n')

	audio_task = asyncio.create_task(process_audio_buffer())

	while True:
		await asyncio.sleep(1)
	audio_task.cancel()
	try:
		await audio_task
	except asyncio.CancelledError:
		print('\nwire was cancelled')

if __name__ == "__main__":
	async def stt_server(websocket, path):
		global latest_data
		while True:
				await websocket.send(latest_data)

	def start_loop(loop, server):
		loop.run_until_complete(server)
		loop.run_forever()

	new_loop = asyncio.new_event_loop()
	start_server = websockets.serve(stt_server, server_host, server_port, loop=new_loop)
	t = Thread(target=start_loop, args=(new_loop, start_server))
	t.kill = Event()
	t.start()
	time.sleep(2)
	print(f"Websocket server started at ws://localhost:{server_port}")

	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		sys.exit('\nInterrupted by user')
		t.kill.set()
		t.join()
