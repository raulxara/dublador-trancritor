import hashlib
import json
from dataclasses import asdict
from decimal import Decimal


class HashDubbingRequestService:
    def exec(self, dto) -> str:
        values = asdict(dto)
        values.pop("actor")
        values.pop("idempotency_key")
        values["speed"] = str(dto.speed.quantize(Decimal("0.01")))
        values["pitch_semitones"] = str(dto.pitch_semitones.quantize(Decimal("0.01")))
        if values["input_text"] is not None:
            values["input_text"] = values["input_text"].strip()
        return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
