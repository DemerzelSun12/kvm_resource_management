#!/bin/bash
vm_domain=$1
maxcpu=$2
virsh destroy ubuntu18.04-$vm_domain
virsh setvcpus ubuntu18.04-$vm_domain --maximum $maxcpu --config
virsh setvcpus ubuntu18.04-$vm_domain --current  $maxcpu
virsh start ubuntu18.04-$vm_domain

