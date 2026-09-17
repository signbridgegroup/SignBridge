import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from move_ugc import MoveUgc
from move_ugc.schemas.job import JobOptions
from move_ugc.schemas.sources import SourceIn


PROJECT_ROOT = Path(r"C:\SignBridge_Project")

ENV_FILE = (
    PROJECT_ROOT
    / "signbridge_integration"
    / "backend"
    / ".env"
)

VIDEO_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sign_videos"
    / "Stack"
    / "Stack_01_1.mov"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "signbridge_integration"
    / "moveai_results"
    / "Stack_01_1_s2"
)

API_ENDPOINT = "https://api.move.ai/ugc/graphql"

REQUESTED_OUTPUTS = [
    "MAIN_FBX",
    "MAIN_GLB",
    "MOTION_DATA",
    "RENDER_OVERLAY_VIDEO",
]


def download_outputs(client, job_id, output_directory):
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    finished_job = client.jobs.retrieve(
        id=job_id,
        expand=["outputs"],
    )

    outputs = finished_job.outputs or []
    downloaded_files = []

    if not outputs:
        print(
            "The job finished, but no downloadable "
            "outputs were returned."
        )
        return downloaded_files

    for index, output in enumerate(outputs, start=1):
        output_file = output.file
        download_url = str(output_file.presigned_url)

        output_name = (
            output.format or f"output_{index}"
        ).lower()

        extension = (
            output_file.type or "bin"
        ).lower().lstrip(".")

        destination = (
            output_directory
            / f"{output_name}.{extension}"
        )

        print(f"Downloading: {output_name}")

        with requests.get(
            download_url,
            stream=True,
            timeout=300,
        ) as response:
            response.raise_for_status()

            with destination.open("wb") as file_handle:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file_handle.write(chunk)

        downloaded_files.append(destination)
        print(f"Saved: {destination}")

    return downloaded_files


def main():
    load_dotenv(
        dotenv_path=ENV_FILE,
        override=True,
    )

    api_key = os.getenv("MOVEAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MOVEAI_API_KEY was not found in: "
            f"{ENV_FILE}"
        )

    if not VIDEO_FILE.is_file():
        raise FileNotFoundError(
            f"Stack video was not found: {VIDEO_FILE}"
        )

    options = JobOptions(
        trackFingers=True,
        floorPlane=True,
        mocapModel="S2",
    )

    print("=" * 60)
    print("SignBridge Move AI test")
    print(f"Input video: {VIDEO_FILE}")
    print(f"Output folder: {OUTPUT_DIRECTORY}")
    print(
        "Options: "
        f"{options.model_dump(by_alias=True)}"
    )
    print(
        "Requested outputs: "
        f"{REQUESTED_OUTPUTS}"
    )
    print(
        "Estimated processing charge: "
        "approximately $0.10-$0.15"
    )
    print("=" * 60)

    confirmation = input(
        "Type YES to upload and start "
        "the paid S2 job: "
    ).strip()

    if confirmation != "YES":
        print(
            "Cancelled. Nothing was uploaded "
            "or charged."
        )
        return

    client = MoveUgc(
        api_key=api_key,
        endpoint_url=API_ENDPOINT,
    )

    print("\nChecking API connection...")
    client.client.retrieve()
    print("API connection successful.")

    print("\nCreating the remote video file...")

    remote_file = client.files.create(
        file_type="mov",
        name="SignBridge_Stack_01_1",
    )

    print(f"Remote file created: {remote_file.id}")
    print("Uploading Stack_01_1.mov...")

    with VIDEO_FILE.open("rb") as video_handle:
        upload_response = requests.put(
            str(remote_file.presigned_url),
            data=video_handle.read(),
            timeout=300,
        )

    upload_response.raise_for_status()
    print("Video upload completed.")

    print("\nCreating a single-camera take...")

    take = client.takes.create_singlecam(
        sources=[
            SourceIn(
                device_label="cam01",
                file_id=remote_file.id,
                format="MOV",
            )
        ],
        name="SignBridge Stack 01 1 S2 Test",
        metadata={
            "project": "SignBridge",
            "sign": "Stack",
            "source": "Stack_01_1.mov",
        },
    )

    print(f"Take created: {take.id}")
    print("\nSubmitting the paid S2 + Dex job...")

    job = client.jobs.create_singlecam(
        take_id=take.id,
        name="SignBridge Stack S2 Dex Test",
        metadata={
            "project": "SignBridge",
            "purpose": "avatar_finger_tracking_test",
        },
        options=options,
        outputs=REQUESTED_OUTPUTS,
    )

    print(f"Job created: {job.id}")
    print("Processing may take several minutes.")

    maximum_attempts = 120
    waiting_seconds = 30

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        current_job = client.jobs.retrieve(
            id=job.id
        )

        state = (
            current_job
            .progress
            .state
            .upper()
        )

        percentage = (
            current_job
            .progress
            .percentage_complete
        )

        print(
            f"Check {attempt}/{maximum_attempts}: "
            f"state={state}, "
            f"progress={percentage}%"
        )

        if state == "FINISHED":
            print(
                "\nProcessing finished. "
                "Downloading outputs..."
            )

            downloaded = download_outputs(
                client,
                job.id,
                OUTPUT_DIRECTORY,
            )

            print("\nMove AI test completed.")
            print(
                f"Downloaded files: "
                f"{len(downloaded)}"
            )

            for path in downloaded:
                print(path)

            return

        if state == "FAILED":
            raise RuntimeError(
                "Move AI processing failed. "
                "Check the API dashboard."
            )

        time.sleep(waiting_seconds)

    raise TimeoutError(
        "The job may still be processing. "
        "Check the Move AI dashboard."
    )


if __name__ == "__main__":
    main()