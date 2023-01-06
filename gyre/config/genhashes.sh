#!/bin/bash
(for x in `git log --pretty=format:"%H" --diff-filter=d --reverse -- ../../engines.yaml` ; do git rev-parse "${x}:../../engines.yaml" ; done) > dist_hashes
(for x in `git log --pretty=format:"%H" --diff-filter=d --reverse -- ./engines.yaml` ; do git rev-parse "${x}:./engines.yaml" ; done) >> dist_hashes

