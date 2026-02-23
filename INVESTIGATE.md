  Patterpunk sends both tools and response_schema to Google's API simultaneously. There's no conflict check, no warning, no merging logic (unlike the Anthropic adapter which converts structured output into a special tool).              
   
  But parsed_output returns None for a different reason than the API rejecting the combination — there are two bugs in the response path:                                                                                                   
                                                                                                                                                                                                                                          
  1. Tool call messages lose structured_output: When Google returns a functionCall part, patterpunk creates a ToolCallMessage with no structured_output attribute. Subsequent complete() calls read structured_output from the latest     
  message (now a ToolResultMessage), which doesn't have it.
  2. Streaming build_final_chat never passes structured_output through: Even when the model's final text response is valid JSON, the AssistantMessage built by the streaming accumulator doesn't include structured_output, so parsed_output
   can't parse it.

  So the API likely did receive response_schema alongside tools — and interestingly, we didn't get a 400 error. That suggests either:
  - Google's API has relaxed the restriction since those GitHub issues were filed
  - Or patterpunk's Google adapter serializes things in a way that avoids the check

  Either way, the model produced plain markdown (not JSON), and even if it had produced JSON, parsed_output would still return None because the streaming path drops the structured_output reference. The Anthropic adapter handles this
  correctly by merging structured output into the tools list — the Google adapter has no equivalent logic.


---

So Anthropic does weird magic to tool call definitions? And Google just doesn't support tool calls at all?!
