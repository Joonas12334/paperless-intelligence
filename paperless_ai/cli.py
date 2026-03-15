import os
import sys

from .config import Settings, get_settings
from .processor import Processor


EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_SETUP_ERROR = 2


def print_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def print_setup_errors(missing_fields: list[str], invalid_fields: list[str]) -> None:
    print("Metadata validation failed.", file=sys.stderr)
    if missing_fields:
        print(f"Missing custom fields: {', '.join(missing_fields)}", file=sys.stderr)
    for issue in invalid_fields:
        print(f"Invalid custom field: {issue}", file=sys.stderr)


def format_ocr_preview(text: str, *, max_lines: int = 12, max_chars: int = 1200) -> str:
    clipped = text[:max_chars].strip()
    lines = clipped.splitlines()
    if len(lines) > max_lines:
        clipped = "\n".join(lines[:max_lines]).rstrip()
    if len(text) > len(clipped):
        clipped += "\n..."
    return clipped


def cmd_bootstrap(settings: Settings) -> int:
    for server in settings.servers:
        print(f"Checking server: {server.name}")
        processor = Processor(settings, server, progress=print_progress)
        ready, missing_fields, invalid_fields, _, _ = processor.bootstrap_validate()
        if not ready:
            print_setup_errors(missing_fields, invalid_fields)
            return EXIT_SETUP_ERROR

        print(f"[{server.name}] Metadata validation passed.")
        print(f"[{server.name}] Required custom field: AI-Processing Status (Queued, Done, Failed)")
    return EXIT_OK


def cmd_run(settings: Settings) -> int:
    total_seen = 0
    total_eligible = 0
    total_processed_ok = 0
    total_failed = 0
    total_skipped = 0

    for server in settings.servers:
        print(f"\n=== Processing server: {server.name} ===")
        processor = Processor(settings, server, progress=print_progress)
        ready, missing_fields, invalid_fields, custom_field_ids, select_option_ids = processor.bootstrap_validate()
        if not ready:
            print_setup_errors(missing_fields, invalid_fields)
            return EXIT_SETUP_ERROR

        summary = processor.run_batch_once(
            custom_field_id=custom_field_ids["status"],
            status_option_ids=select_option_ids.get("status", {}),
        )
        total_seen += summary.seen
        total_eligible += summary.eligible
        total_processed_ok += summary.processed_ok
        total_failed += summary.failed
        total_skipped += summary.skipped

        print(f"[{server.name}] Batch complete.")
        print(f"[{server.name}] seen={summary.seen}")
        print(f"[{server.name}] eligible={summary.eligible}")
        print(f"[{server.name}] processed_ok={summary.processed_ok}")
        print(f"[{server.name}] failed={summary.failed}")
        print(f"[{server.name}] skipped={summary.skipped}")

    print(f"\n=== Overall Summary ===")
    print(f"Total seen={total_seen}")
    print(f"Total eligible={total_eligible}")
    print(f"Total processed_ok={total_processed_ok}")
    print(f"Total failed={total_failed}")
    print(f"Total skipped={total_skipped}")
    return EXIT_OK


def cmd_preview(settings: Settings, document_id: int, server_name: str | None = None) -> int:
    if document_id < 1:
        print("document_id must be >= 1", file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    if server_name:
        servers = [s for s in settings.servers if s.name == server_name]
        if not servers:
            print(f"Server '{server_name}' not found", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
    else:
        servers = settings.servers
        if len(servers) > 1:
            print("Multiple servers configured. Use --server to specify which to use.", file=sys.stderr)
            for s in servers:
                print(f"  - {s.name}")
            return EXIT_RUNTIME_ERROR

    server = servers[0]
    processor = Processor(settings, server, progress=print_progress)
    ready, missing_fields, invalid_fields, custom_field_ids, select_option_ids = processor.bootstrap_validate()
    if not ready:
        print_setup_errors(missing_fields, invalid_fields)
        return EXIT_SETUP_ERROR

    preview = processor.preview_document(document_id)

    print(f"Server: {server.name}")
    print(f"Document: #{preview.document_id}")
    print(f"Current title: {preview.current_title}")
    print(f"Proposed title: {preview.proposed_title}")
    print(f"Language: {preview.language or 'unknown'}")
    print(f"Processing mode: {preview.processing_mode}")
    if preview.pdf_mode:
        print(f"PDF mode: {preview.pdf_mode}")
    print(f"Detection: {preview.detection_reason}")
    print(f"Will overwrite content: {'yes' if preview.overwrite_content else 'no'}")

    confidence = preview.confidence
    if isinstance(confidence, float):
        print(f"Confidence: {confidence:.2f}")
    else:
        print("Confidence: n/a")
    print("Content preview:")
    print(format_ocr_preview(preview.content_preview))

    answer = input("Save title to Paperless? [y/N]: ").strip().lower()
    if answer in {"y", "yes"}:
        saved = processor.save_preview(
            document_id,
            preview.proposed_title,
            preview.content_preview,
            custom_field_id=custom_field_ids["status"],
            status_option_ids=select_option_ids.get("status", {}),
            overwrite_content=bool(preview.overwrite_content),
        )
        print(f"Saved title: {saved}")
    else:
        print("Cancelled.")
    return EXIT_OK


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def main_interactive() -> int:
    settings = None
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Warning: Could not load settings: {e}")
        print("Edit .env file to configure.\n")

    while True:
        clear_screen()
        print("=== Paperless AI ===")
        print("1. Run batch (process all servers)")
        print("2. Preview document")
        print("3. Validate config")
        print("4. Exit")
        print()

        choice = input("Select option: ").strip()

        if choice == "1":
            if settings is None:
                try:
                    settings = get_settings()
                except Exception as e:
                    print(f"Failed to load settings: {e}")
                    input("\nPress Enter to continue...")
                    continue
            clear_screen()
            print("=== Running Batch ===\n")
            cmd_run(settings)
            input("\nPress Enter to continue...")

        elif choice == "2":
            if settings is None:
                try:
                    settings = get_settings()
                except Exception as e:
                    print(f"Failed to load settings: {e}")
                    input("\nPress Enter to continue...")
                    continue

            doc_id_input = input("Enter document ID: ").strip()
            if not doc_id_input.isdigit():
                print("Invalid document ID")
                input("\nPress Enter to continue...")
                continue

            document_id = int(doc_id_input)
            server_name = None
            if len(settings.servers) > 1:
                print("\nAvailable servers:")
                for i, s in enumerate(settings.servers, 1):
                    print(f"  {i}. {s.name}")
                server_choice = input("Select server (or press Enter for first): ").strip()
                if server_choice.isdigit() and 1 <= int(server_choice) <= len(settings.servers):
                    server_name = settings.servers[int(server_choice) - 1].name

            clear_screen()
            print(f"=== Preview Document #{document_id} ===\n")
            cmd_preview(settings, document_id, server_name)
            input("\nPress Enter to continue...")

        elif choice == "3":
            if settings is None:
                try:
                    settings = get_settings()
                except Exception as e:
                    print(f"Failed to load settings: {e}")
                    input("\nPress Enter to continue...")
                    continue
            clear_screen()
            print("=== Bootstrap ===\n")
            result = cmd_bootstrap(settings)
            if result == EXIT_OK:
                print("\nAll servers validated successfully!")
            input("\nPress Enter to continue...")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid option")
            input("\nPress Enter to continue...")

    return EXIT_OK


def main() -> int:
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command in ("-h", "--help"):
            print("Usage: python paperless-ai.py [command]")
            print("Commands: validate, run, preview <id>")
            print("Or run without arguments for interactive menu.")
            return EXIT_OK

        settings = get_settings()

        if command == "validate":
            return cmd_bootstrap(settings)
        elif command == "run":
            return cmd_run(settings)
        elif command == "preview":
            if len(sys.argv) < 3:
                print("preview requires document_id", file=sys.stderr)
                return EXIT_RUNTIME_ERROR
            doc_id = int(sys.argv[2])
            server = None
            for i, arg in enumerate(sys.argv):
                if arg in ("-s", "--server") and i + 1 < len(sys.argv):
                    server = sys.argv[i + 1]
            return cmd_preview(settings, doc_id, server)
        else:
            print(f"Unknown command: {command}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
    else:
        return main_interactive()


if __name__ == "__main__":
    sys.exit(main())
