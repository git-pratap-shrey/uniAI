# Email Forensics

**Subject**: CYBER_SECURITY | **Unit**: unit4

## Definition
Email forensics is the systematic examination and investigation of email content, headers, and attachments to identify the sender, recipient, and transmission path for legal purposes.

## Key points
- Header Analysis: Provides the path of the email; includes IP addresses and server information.
- Message-ID: A globally unique identifier used to track email logs across servers.
- X-Originating-IP: Used to trace the actual IP address of the sender's computer.
- Server Investigation: If emails are deleted from clients, ISP or proxy server logs are scanned.
- Spoofing: Headers can be easily spoofed; investigators must verify server logs to confirm authenticity.

## Important terms
| Term | Meaning |
|------|---------|
| Message-ID | Unique string identifying an email and its version. |
| MIME | Internet standard extending message formats. |
| DKIM-Signature | Cryptographic authentication for email domains. |

## Exam questions likely on this topic
- Explain the importance of email header analysis in forensics.
- How can an investigator trace the source of an email if the client logs are deleted?

## See also
- [[Network_Forensics]]
