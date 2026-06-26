from mcp.server.fastmcp import FastMCP
from timesheet import timesheet
from datetime import datetime

mcp = FastMCP(
    "customer-service", 
    host="0.0.0.0",    
    port=4040,         
    log_level="DEBUG"  
)


@mcp.tool()
def get_current_date_time() -> str:
    """Get current datetime in "%Y-%m-%d %H:%M:%S.%f" format."""
    return str(datetime.now())

@mcp.tool()
def execute_user_commands(command_string:str)->str:
    """
    Execute a registered user commands.
    Args:command_string: The full command string.
    Returns:The command response.
    Examples:
        > execute_user_command("#timesheet")
        "Today's timesheet..."

        > execute_user_command("#timesheet --showmore")
        "Showing complete timesheet..."
        > execute_user_command("#unknown")
        "Unknown user command: #unknown"
    """
    print('CALLED---',command_string)
    commands=['#timesheet','#time']
    user_command=command_string.split(' ')[0]
    if user_command in commands:
        if user_command == '#timesheet':
            return timesheet(command_string)
        elif user_command == '#time':
            return timesheet(command_string)
    else:
        return 'Unknown user command'

if __name__ == "__main__":
    mcp.run(transport="sse")
   