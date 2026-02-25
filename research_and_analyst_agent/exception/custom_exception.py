
import sys
import traceback
from typing import Optional, cast

class ResearchAnalystException(Exception):
    def __init__(self, error_message, error_details: Optional[object] = None):
        
        # Normalize message
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # Resolve exc_info (supports: sys module, Exception object, or current context)
        exc_type = exc_value = exc_tb = None
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            if hasattr(error_details, "exc_info"):  # e.g., sys
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk to the last frame to report the most relevant location
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Full pretty traceback (if available)
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        # Compact, logger-friendly message (no leading spaces)
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return f"ResearchAnalystException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"


if __name__ == "__main__":
    # Demo-1: generic exception -> wrap
    try:
        a = 1 / 0
    except Exception as e:
        raise ResearchAnalystException("Division failed", e) from e

    # Demo-2: still supports sys (old pattern)
    # try:
    #     a = int("abc")
    # except Exception as e:
    #     raise ResearchAnalystException(e, sys)
    
    
    # BaseException
    # ├── Exception
    # │     ├── ValueError
    # │     ├── TypeError
    # │     ├── KeyError
    # │     └── ...
    # ├── SystemExit
    # ├── KeyboardInterrupt

# validation error tha?
# config missing tha?
# LLM API down tha?
# embedding fail hua?
# vector DB fail hua?

# 👉 Matlab: unexpected / unhandled errors ko wrap karke clean error banana.


# Create a Python base custom exception class for a production AI/LLM pipeline that:
# Captures file name and line number from traceback
# Supports wrapping original exceptions
# Extracts full traceback string
# Accepts optional context dictionary (like user_id, model_name, db_name)
# Formats error message in a compact, logger-friendly format
# Includes str and repr methods