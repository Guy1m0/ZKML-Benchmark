#! /usr/bin/env node
const { execSync } = require('child_process');
// Increase memory limit to 4 GB, adjust as needed
const maxOldSpaceSize = 65536; 
const snarkjsCommand = `node --max-old-space-size=${maxOldSpaceSize} $(which snarkjs) ${process.argv.slice(2).join(' ')}`;

try {
    const output = execSync(snarkjsCommand, { stdio: 'inherit' });
    console.log(output.toString());
} catch (error) {
    console.error(`Error executing snarkjs: ${error}`);
    process.exit(1);
}
