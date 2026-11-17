#!/bin/sh
# Build the reduced Java runtime that the packaged app carries (build/jre), from a JDK 21 or
# newer: JAVA_HOME if set, else the `java` on the PATH. Run from the repository root.
set -eu

JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")}"
rm -rf build/jre
"$JAVA_HOME/bin/jlink" \
    --add-modules java.base,java.logging,java.xml,java.management,java.naming,java.prefs,java.sql,java.desktop,jdk.unsupported,jdk.zipfs,jdk.crypto.ec \
    --strip-debug --no-header-files --no-man-pages --compress=zip-6 \
    --output build/jre
