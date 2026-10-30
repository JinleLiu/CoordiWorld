#!/usr/bin/env python3
"""Reassemble the unchanged reference-world checkpoint from ordered transport parts."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
PARTS=ROOT/'models/world/parts'

def assemble():
    manifest=json.loads((PARTS/'manifest.json').read_text())
    target=ROOT/'models/world'/manifest['target']
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==manifest['sha256']:
        return target
    content=[]
    for record in manifest['parts']:
        data=(PARTS/record['path']).read_bytes()
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('Checkpoint part mismatch: '+record['path'])
        content.append(data)
    data=b''.join(content)
    if len(data)!=manifest['bytes'] or hashlib.sha256(data).hexdigest()!=manifest['sha256']:
        raise ValueError('Reassembled checkpoint does not match its frozen identity.')
    target.write_bytes(data)
    return target

if __name__=='__main__':
    path=assemble()
    print('Reassembled frozen checkpoint:',path.name)
