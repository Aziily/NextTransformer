// Every external address the site links to.
export const REPO_URL = 'https://github.com/Interactive-Active-AI/NextTransformer';
export const REPO_API = 'https://api.github.com/repos/Interactive-Active-AI/NextTransformer';
export const REPO_FILE = REPO_URL+'/blob/main/';
export const DISCUSSIONS_URL = REPO_URL+'/discussions';
// Permanent invite to the public #general channel.
export const DISCORD_URL = 'https://discord.gg/k2XPXW6Jz';
const BASE = import.meta.env.BASE_URL;
export const asset = (path: string) => BASE+path.replace(/^\//, '');
// The documentation is deployed next to the site, under docs/.
export const docs = (page = '') => BASE+'docs/'+page;
