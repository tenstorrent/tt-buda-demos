# FAQ & Troubleshooting Guide

## Resetting an accelerator board

If you encounter a situation where a Tenstorrent chip appears to be unresponsive or is exhibiting unusual behavior, a software reset of the board might be a viable solution.


For a software reset on a single chip use : `tt-smi -lr 0` .

For more information on performing reset on multiple chips or other specifics visit [TT-SMI Resets](https://github.com/tenstorrent/tt-smi?tab=readme-ov-file#resets:~:text=on%20the%20footer.-,Resets,-Another%20feature%20of)

If you need additional assistance, you can access a detailed explanation of all available command options by appending the help flag to the command like so: `tt-smi --help` or `tt-smi -h`.

For comprehensive insights and detailed instructions on utilizing the command line GUI, we invite you to explore the Tenstorrent System Management Interface (TT-SMI) repository on GitHub at [tt-smi-repo](https://github.com/tenstorrent/tt-smi). TT-SMI serves as a versatile command-line utility tailored to streamline interaction with all Tenstorrent devices on host.

If the software reset fails to resolve the issue, the next step would be to power cycle the board. This typically involves rebooting the host machine that the Tenstorrent board is connected to. 
*Please note that any unsaved work may be lost during this process, so ensure all important data is saved before proceeding*

## `PermissionError` on `/tmp/*.lock` files

If multiple users are running on a system with a shared Tenstorrent device,
you may encounter a `PermissionError: [Errno 13] Permission denied: '/tmp/*.lock'`.

You would need to remove these files between user sessions.
