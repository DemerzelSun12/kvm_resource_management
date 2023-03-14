#!/usr/bin/expect
spawn ssh root@192.168.122.22
expect "*password:"
send "vm1\r"
expect "*#"
interact
